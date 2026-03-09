import types, time, torch
from transformers import GenerationConfig
import contextlib
from transformers.cache_utils import StaticCache
from contextlib import nullcontext as _nullctx

# (wjbang, 2026.03.04)
# Although this function is written for models with Triton kernels,
# it can be used for any models, including models with custom CUDA kernels (e.g. AQLM's CUDA kernels).
# Just calibrate the warmp_iters to adequate level when using with non-Triton kernels.
def attach_cudagraph_generate_triton(model, tokenizer,
                                     device: str = "cuda:0",
                                     dtype: torch.dtype = torch.float16,
                                     warmup_iters: int = 5):
    """
    CUDA-graph accelerated generate() for models whose linear layers use
    **Triton kernels** (e.g. AQLM QuantizedLinear falling back to
    ``triton_matmul`` via ``@triton.autotune``).

    Why a separate version?
    -----------------------
    ``attach_cudagraph_generate`` wraps the decode step with
    ``torch.compile(fullgraph=True, mode="reduce-overhead")`` and then
    captures it inside an explicit ``torch.cuda.CUDAGraph``.  This fails or
    produces incorrect graphs when the model contains Triton-autotuned
    kernels for two reasons:

    1. **torch.compile + custom autograd.Function**
       ``QuantizedLinear`` dispatches through ``_QuantizedMatmul``, a
       ``torch.autograd.Function`` subclass.  ``fullgraph=True`` cannot
       trace through ``Function.apply()`` and will either error out or
       silently fall back, defeating the purpose of compilation.

    2. **Triton @autotune inside CUDA-graph capture**
       On the very first call, ``@triton.autotune`` benchmarks every
       candidate config (num_stages x num_warps grid — 20 variants in the
       current AQLM triton kernel).  If that exploration happens *inside*
       ``torch.cuda.graph()``, every benchmarking launch is recorded into
       the graph, and ``g.replay()`` will faithfully re-execute all of them
       on every decode step — producing garbage output and ~20x overhead.

    This version fixes both issues:
      • **No torch.compile** — the decode function is kept as-is.
      • **Explicit warmup** — ``warmup_iters`` eager forward passes run
        *before* graph capture, guaranteeing that Triton JIT compilation
        and autotune have fully converged.  Only the final, optimal kernel
        is recorded into the CUDA graph.

    Everything else (``StaticCache``, stub-tensor mutation, eager prefill,
    greedy decode loop) is identical to the original helper.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal-LM already on *device*, with QuantizedLinear
        layers that route to Triton kernels.
    tokenizer : PreTrainedTokenizer
        Matching tokenizer (used only for pad-token config).
    device : str
        CUDA device string, e.g. ``"cuda:0"``.
    dtype : torch.dtype
        Compute dtype for the static KV cache (should match model dtype).
    warmup_iters : int
        Number of eager forward passes executed before graph capture.
        Must be ≥ 1 (to trigger Triton compile + autotune).
        3-5 is recommended; the cost is a one-time overhead per distinct
        ``(batch_size, max_cache_len)`` pair.
    """

    assert warmup_iters >= 1, (
        "warmup_iters must be >= 1 to trigger Triton JIT compilation and autotune"
    )

    # ------------------------------------------------------------------ helpers
    def _build_graph(batch_size: int, max_cache_len: int):
        """
        Warm up Triton kernels and then capture the 1-token decode step
        into a CUDA graph for ``(batch_size, max_cache_len)``.
        """
        nonlocal device, dtype

        device_t = torch.device(device)

        # 1) Static cache (shape must never change after capture)
        static_cache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device_t,
            dtype=dtype,
        )

        # 2) Stub tensors — mutated in-place per replay
        stub_ids  = torch.empty((batch_size, 1), dtype=torch.long, device=device_t)
        stub_pos  = torch.empty_like(stub_ids)
        stub_cp   = torch.empty((1,), dtype=torch.long, device=device_t)
        stub_mask = torch.zeros(
            (batch_size, max_cache_len), dtype=torch.bool, device=device_t
        )

        # 3) Raw one-token decode function — NO torch.compile
        #    torch.compile with fullgraph=True is incompatible with custom
        #    autograd.Function (QuantizedLinear uses _QuantizedMatmul).
        #    mode="reduce-overhead" would also double-capture CUDA graphs.
        def _decode_one(input_ids, position_ids,
                        attention_mask, past_key_values, cache_position):
            return model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            ).logits

        # 4) ── Warmup ──────────────────────────────────────────────────
        #    Run several eager forward passes so that:
        #      • Triton JIT compiles every unique kernel signature
        #      • @triton.autotune benchmarks all candidate configs and
        #        caches the winner
        #      • cuBLAS / cuDNN plan caches are populated
        #      • QuantizedLinear.prepare_matmul_op() is called once,
        #        selecting the gemv / gemm op for the decode input shape
        #
        #    After this loop only the optimal kernel variant will be
        #    launched, making it safe to capture into a CUDA graph.
        stub_ids.fill_(0)
        stub_pos.fill_(0)
        stub_cp.fill_(0)
        stub_mask[:, 0] = True

        with torch.inference_mode():
            for _ in range(warmup_iters):
                static_cache.reset()
                _decode_one(
                    stub_ids, stub_pos, stub_mask, static_cache, stub_cp
                )
        torch.cuda.synchronize()

        # 5) ── Capture ─────────────────────────────────────────────────
        #    Reset all mutable state so the graph starts from a clean
        #    slate identical to what cg_generate will provide at runtime.
        static_cache.reset()
        stub_mask.zero_()

        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            logits_buf = _decode_one(
                stub_ids, stub_pos, stub_mask, static_cache, stub_cp
            )

        return dict(
            g=g, logits_buf=logits_buf, static_cache=static_cache,
            stub_ids=stub_ids, stub_pos=stub_pos,
            stub_cp=stub_cp, stub_mask=stub_mask,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
        )

    # ---------------------------------------------------------------- override
    def cg_generate(self,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor = None,
                    max_new_tokens: int = 20,
                    do_sample: bool = False,
                    **kwargs):
        assert do_sample is False, "graph-generate only supports greedy decoding"
        if max_new_tokens <= 0:
            return input_ids

        device_t = input_ids.device
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        max_cache_len = prompt_len + max_new_tokens

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=device_t
            )
        else:
            attention_mask = attention_mask.to(device_t)

        prompt_lengths = attention_mask.long().sum(dim=1)
        if torch.any(prompt_lengths <= 0):
            raise ValueError(
                "All prompts must contain at least one non-padding token."
            )

        # Lazily build / retrieve the graph for this (batch, len) pair
        if not hasattr(self, "_cg_graphs_triton"):
            self._cg_graphs_triton = {}

        cg_key = (batch_size, max_cache_len)
        if cg_key not in self._cg_graphs_triton:
            self._cg_graphs_triton[cg_key] = _build_graph(
                batch_size, max_cache_len
            )

        cg           = self._cg_graphs_triton[cg_key]
        static_cache = cg["static_cache"]
        stub_ids     = cg["stub_ids"]
        stub_pos     = cg["stub_pos"]
        stub_cp      = cg["stub_cp"]
        stub_mask    = cg["stub_mask"]
        g            = cg["g"]
        logits_buf   = cg["logits_buf"]

        # ---- 0) reset cache & mask ----------------------------------
        static_cache.reset()
        stub_mask.zero_()
        stub_mask[:, :prompt_len] = attention_mask.to(torch.bool)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # ---- 1) PREFILL (runs eagerly) ------------------------------
        with torch.inference_mode():
            p_out = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=static_cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=device_t),
                # logits_to_keep=1 # added, tempfix
            )

            batch_idx   = torch.arange(batch_size, device=device_t) # removed, tempfix
            last_pos    = (prompt_lengths - 1).clamp(min=0) # removed, tempfix
            last_logits = p_out.logits[batch_idx, last_pos, :] # removed, tempfix
            # last_logits = p_out.logits[:, -1, :] # added, tempfix
            next_token  = last_logits.argmax(-1, keepdim=True)

        generated = [input_ids, next_token]

        # ---- 2) TOKEN-BY-TOKEN decode (CUDA graph replay) -----------
        next_pos_ids = prompt_lengths.view(-1, 1).long()
        with torch.inference_mode():
            for step in range(max_new_tokens - 1):
                curr_cache_pos = prompt_len + step

                # mutate stubs *in-place* for the graph
                stub_ids.copy_(next_token)
                stub_pos.copy_(next_pos_ids)
                stub_cp.fill_(curr_cache_pos)
                stub_mask[:, curr_cache_pos] = True

                # single-launch decode
                g.replay()

                # logits_buf is updated in-place by the replay
                next_token = logits_buf[:, -1].argmax(-1, keepdim=True)
                generated.append(next_token)
                next_pos_ids.add_(1)

        return torch.cat(generated, dim=1)

    # ---------------------------------------------------------------- attach!
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = "static"

    model.generate = types.MethodType(cg_generate, model)

def attach_cudagraph_generate(model, tokenizer,
                              device: str = "cuda:0",
                              dtype: torch.dtype = torch.float16):
    """
    Replaces `model.generate` with a CUDA-graph version that
    (1) runs the prompt in eager mode (prefill) and
    (2) replays a captured 1-token decode graph for every step.
    
    ⚠️  Designed for:
        • batch_size == 1
        • greedy decoding (do_sample = False)
        • max_new_tokens ≤ value passed at runtime
    """

    # ------------------------------------------------------------------ helpers
    def _build_graph(max_cache_len: int):
        """Capture the 1-token decode graph for the current `max_cache_len`."""
        nonlocal device, dtype

        # 1.  Static cache (shape must never change after capture)
        static_cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=torch.device(device),
            dtype=dtype,
        )

        # 2.  Stub tensors whose *data* we mutate each replay
        stub_ids  = torch.empty((1, 1),  dtype=torch.long,   device=device)
        stub_pos  = torch.empty_like(stub_ids)
        stub_cp   = torch.empty((1,),   dtype=torch.long,   device=device)
        stub_mask = torch.zeros((1, max_cache_len), dtype=torch.bool, device=device)

        # 3.  A tiny wrapper that does one-token decoding
        def _decode_one(input_ids, position_ids,
                        attention_mask, past_key_values, cache_position):
            return model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            ).logits

        _decode_one = torch.compile(               # <-- optional, helps perf
            _decode_one, fullgraph=True,
            mode="reduce-overhead",
            disable={"cudagraphs_rng", "cudagraphs_dropout"}
        )
        
        # ---------- NEW: PRIME ALL LAZY KERNELS (cuBLAS, Triton, etc.) -----
        torch.empty((8, 8), device=device, dtype=torch.float32) @ \
        torch.empty((8, 8), device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        # -------------------------------------------------------------------

        # 4.  Capture!
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits_buf = _decode_one(
                stub_ids, stub_pos, stub_mask, static_cache, stub_cp
            )

        return dict(
            g=g, logits_buf=logits_buf, static_cache=static_cache,
            stub_ids=stub_ids, stub_pos=stub_pos,
            stub_cp=stub_cp, stub_mask=stub_mask,
            max_cache_len=max_cache_len
        )

    # ---------------------------------------------------------------- override
    def cg_generate(self,
                    input_ids: torch.Tensor,
                    max_new_tokens: int = 20,
                    do_sample: bool = False,
                    **kwargs):
        assert do_sample is False, "graph‐generate only supports greedy decoding"
        assert input_ids.shape[0] == 1, "only batch_size==1 is supported"

        prompt_len     = input_ids.shape[1]
        max_cache_len  = prompt_len + max_new_tokens

        # (Re)build graph if prompt is longer than previous capture
        if (not hasattr(self, "_cg") or
                self._cg["max_cache_len"] < max_cache_len):
            self._cg = _build_graph(max_cache_len)

        cg               = self._cg
        static_cache     = cg["static_cache"]
        stub_ids         = cg["stub_ids"]
        stub_pos         = cg["stub_pos"]
        stub_cp          = cg["stub_cp"]
        stub_mask        = cg["stub_mask"]
        g                = cg["g"]
        logits_buf       = cg["logits_buf"]

        # ---- 0) reset cache & mask ------------------------------------------
        static_cache.reset()                    # zero-out keys/values in-place
        stub_mask.zero_()
        stub_mask[:, :prompt_len] = True

        # ---- 1) PREFILL (runs eagerly) --------------------------------------
        with torch.inference_mode():
            p_out = self(                       # fills `static_cache`
                input_ids=input_ids,
                past_key_values=static_cache,
                use_cache=True,
                cache_position=torch.arange(prompt_len, device=device)
            )
            next_token = p_out.logits[:, -1].argmax(-1, keepdim=True)

        generated = [input_ids, next_token]

        # ---- 2) TOKEN-BY-TOKEN decode (CUDA graph replay) -------------------
        for step in range(max_new_tokens - 1):
            curr_pos = prompt_len + step

            # mutate stubs *in-place* for the graph
            stub_ids.copy_(next_token)
            stub_pos.fill_(curr_pos)
            stub_cp.fill_(curr_pos)
            stub_mask[:, curr_pos] = True

            # single-launch decode
            g.replay()

            # logits_buf is updated in-place by the replay
            next_token = logits_buf[:, -1].argmax(-1, keepdim=True)
            generated.append(next_token)

        return torch.cat(generated, dim=1)

    # ---------------------------------------------------------------- attach!
    # Ensure GenerationConfig uses static caching so HF's `generate()` doesn’t
    # try to re-allocate DynamicCache internally.
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = "static"

    # Monkey-patch
    model.generate = types.MethodType(cg_generate, model)
    

def _make_cg_runner(fn, example_inp):
    """
    Capture `fn(example_inp)` into a CUDA graph and return a wrapper
    that takes a *tensor with the same shape/dtype* and returns the
    graph's output buffer.

    ──  How it works  ───────────────────────────────────────────────
      • create a stub copy of the example input
      • allocate an output buffer with the correct shape
      • capture  fn(stub)  once inside  torch.cuda.CUDAGraph()
      • each call:
            stub.copy_(x)   # GPU-side memcpy
            graph.replay()  # single-launch execution
            return out_buf
    """
    device = example_inp.device
    dtype  = example_inp.dtype

    # ------- 1) allocate stubs & prime kernels ---------------------
    stub_in  = torch.empty_like(example_inp, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(3):
            ref_out = fn(example_inp)
    out_buf = torch.empty_like(ref_out, device=device, dtype=ref_out.dtype)

    # tiny GEMM to make sure cuBLAS / Triton has loaded its kernels
    (torch.empty((8, 8), device=device) @ torch.empty((8, 8), device=device))
    torch.cuda.synchronize()

    # ------- 2) graph capture -------------------------------------
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g):
        out_buf.copy_(fn(stub_in))

    # ------- 3) wrapped callable ----------------------------------
    def cg_fn(x):
        stub_in.copy_(x)   # mutate input *inside* static memory
        g.replay()         # <1 µs CPU side
        return out_buf

    return cg_fn

@torch.inference_mode()
def validate_cg_runner():
    import torch

    torch.manual_seed(42)
    device = "cuda"

    # ── 1) matmul 기반 fn ──────────────────────────────────────
    W = torch.randn(128, 64, device=device, dtype=torch.float16)

    def fn(x):
        return x @ W

    # ── 2) cg_runner 생성 ──────────────────────────────────────
    example_inp = torch.randn(32, 128, device=device, dtype=torch.float16)
    cg_fn = _make_cg_runner(fn, example_inp)

    # ── 3) 여러 입력에 대해 eager vs graph 비교 + aliasing 검증 ─
    num_tests = 5
    prev_cloned = None  # 직전 graph 결과의 clone

    for i in range(num_tests):
        x = torch.randn_like(example_inp)
        eager_out = fn(x)
        graph_out = cg_fn(x)          # out_buf 참조

        assert eager_out.shape == graph_out.shape, \
            f"[test {i}] shape mismatch: {eager_out.shape} vs {graph_out.shape}"
        assert eager_out.dtype == graph_out.dtype, \
            f"[test {i}] dtype mismatch: {eager_out.dtype} vs {graph_out.dtype}"

        max_diff = (eager_out - graph_out).abs().max().item()
        print(f"  [test {i}] max |eager − graph| = {max_diff:.5f}")

        # aliasing 검증: 이전 clone이 현재 replay로 오염되지 않았는지
        if prev_cloned is not None:
            prev_eager = fn(prev_x)
            alias_diff = (prev_cloned - prev_eager).abs().max().item()
            print(f"          aliasing check (prev clone vs prev eager) = {alias_diff:.5f}")

        prev_cloned = graph_out.clone()
        prev_x = x.clone()

    # ── 4) shape mismatch → RuntimeError 확인 ──────────────────
    wrong_shape = torch.randn(16, 128, device=device, dtype=torch.float16)
    try:
        cg_fn(wrong_shape)
        print("⚠️  shape mismatch가 에러 없이 통과됨 — 주의 필요")
    except RuntimeError:
        print("✅ shape mismatch 시 RuntimeError 정상 발생")

    print(f"✅ validate_cg_runner done — {num_tests} tests")
    
if __name__ == "__main__":
    validate_cg_runner()