# SBVR Kernel Profiling with Nsight Compute

이 디렉토리는 SBVR CUDA 커널을 Nsight Compute (NCU)를 사용하여 프로파일링하기 위한 도구들을 포함합니다.

## 지원 커널

1. **`fused_rtn7_lut_bvr`** (및 `fused_rtn5_lut_bvr`, `fused_rtn3_lut_bvr`)
   - RTN 양자화 + LUT 리맵 + 8×32-bit BVR 패킹을 수행하는 융합 커널
   - 그룹 단위 max-abs 리덕션 → 양자화 → 비트 벡터 패킹
   - 입력: `x` (half, [K]) - 1D 입력 벡터
   - 출력: `out_bvr` (uint32), `scales` (float)

2. **`rtn_N_sbvr_1xtN_mm_T`** (N = 7, 5, 3)
   - RTN-SBVR 행렬 곱셈 커널
   - RTN 양자화된 입력과 SBVR 양자화된 가중치 간의 행렬 곱
   - 입력: `l_bvr`, `l_scales`, `r_bvr`, `r_coeff_idx`, `r_coeff_cache`, `bias`
   - 출력: `out` (half, [1, N])

## 파일 구조

```
nc_kernel_evals/
├── README.md                 # 이 문서
├── eval_sbvr_kernels.py      # 메인 프로파일링 스크립트
├── kernel_runner.py          # 커널 실행 유틸리티 (NCU용)
├── run_ncu_profile.sh        # NCU 프로파일링 쉘 스크립트
└── ncu_reports/              # NCU 리포트 출력 디렉토리 (자동 생성)
```

## 사전 요구사항

1. **SBVR 패키지 설치**:
   ```bash
   cd /home/wjbang/workspace/sbvr
   pip install -e .
   ```

2. **CUDA Toolkit** (NCU 포함):
   - Nsight Compute CLI (`ncu`)가 PATH에 있어야 함
   - NCU GUI (`ncu-ui`)는 리포트 분석에 사용

3. **Root/Sudo 권한** (NCU 전체 메트릭 수집 시 필요)

## 사용법

### 1. 벤치마크 모드 (빠른 타이밍 측정)

CUDA 이벤트를 사용한 빠른 성능 측정:

```bash
# 모든 커널, 기본 설정으로 벤치마크
python eval_sbvr_kernels.py --mode benchmark --kernel all

# 특정 커널만 벤치마크
python eval_sbvr_kernels.py --mode benchmark --kernel rtn_sbvr_mm

# Llama-7B FFN 설정으로 벤치마크
python eval_sbvr_kernels.py --mode benchmark --kernel rtn_sbvr_mm --config llama_7b_ffn_up

# 여러 nRTN 값 테스트
python eval_sbvr_kernels.py --mode benchmark --kernel all --nRTN 7 5 3

# 결과를 JSON으로 저장
python eval_sbvr_kernels.py --mode benchmark --kernel all --save-results results.json
```

### 2. NCU 프로파일링 - Python 스크립트

NCU 명령어 생성 및 실행:

```bash
# NCU 명령어 생성 (출력만)
python eval_sbvr_kernels.py --mode ncu --kernel rtn_sbvr_mm --dry-run

# NCU 프로파일링 실행 (sudo 필요)
sudo python eval_sbvr_kernels.py --mode ncu --kernel all --run-ncu

# Roofline 분석
sudo python eval_sbvr_kernels.py --mode ncu --kernel rtn_sbvr_mm --metrics-set roofline --run-ncu
```

### 3. NCU 프로파일링 - 쉘 스크립트 (권장)

더 편리한 쉘 스크립트 사용:

```bash
# 도움말
./run_ncu_profile.sh --help

# 사용 가능한 설정 목록
./run_ncu_profile.sh --list-configs

# rtn_sbvr_mm 커널 프로파일링
./run_ncu_profile.sh --kernel rtn_sbvr_mm --config medium

# 빠른 프로파일링 (summary 메트릭만)
./run_ncu_profile.sh --kernel all --quick

# 전체 메트릭 수집 (sudo 필요)
sudo ./run_ncu_profile.sh --kernel rtn_sbvr_mm --full-metrics

# Roofline 분석
sudo ./run_ncu_profile.sh --kernel rtn_sbvr_mm --roofline

# 벤치마크 모드 (NCU 없이 타이밍만)
./run_ncu_profile.sh --benchmark --kernel all
```

### 4. 직접 커널 실행 (NCU 수동 프로파일링)

```bash
# 단일 커널 호출 (NCU가 후킹하기 위한)
python kernel_runner.py --kernel fused_rtn_lut_bvr --K 4096

python kernel_runner.py --kernel rtn_sbvr_mm --N 4096 --K 4096 --nRTN 7

python kernel_runner.py --kernel fused_rtn_sbvr_mm --N 4096 --K 4096 --nRTN 7
```

## 테스트 설정

| 설정 이름 | N | K | 설명 |
|-----------|------|------|------|
| small | 2048 | 2048 | 작은 테스트 |
| medium | 4096 | 4096 | 중간 크기 (기본) |
| large | 8192 | 8192 | 큰 크기 |
| llama_7b_attn | 4096 | 4096 | Llama-7B Attention projection |
| llama_7b_ffn_up | 11008 | 4096 | Llama-7B FFN up projection |
| llama_7b_ffn_down | 4096 | 11008 | Llama-7B FFN down projection |

## NCU 리포트 분석

### GUI에서 보기 (권장)

```bash
ncu-ui ./ncu_reports/rtn_sbvr_mm_medium_nRTN7.ncu-rep
```

### CSV로 내보내기

```bash
ncu --import ./ncu_reports/rtn_sbvr_mm_medium_nRTN7.ncu-rep --csv > analysis.csv
```

### 콘솔에서 요약 보기

```bash
ncu --import ./ncu_reports/rtn_sbvr_mm_medium_nRTN7.ncu-rep --page details
```

## 주요 분석 메트릭

### fused_rtn7_lut_bvr 커널

- **메모리 대역폭 활용도**: 입력 로드와 BVR/scales 저장 효율
- **공유 메모리 뱅크 충돌**: warp 내 max 리덕션 시 확인
- **메모리 접근 패턴**: Coalesced 접근 여부

### rtn_sbvr_1xtN_mm_T 커널

- **명령어 처리량**: `__popc` 연산 활용도
- **메모리/연산 밸런스**: Memory bound vs Compute bound
- **레지스터 압력**: 스필링 여부
- **Occupancy**: SM 활용도

## 커널 입출력 상세

### fused_rtn7_lut_bvr

```cpp
// 입력
const __half* x        // [K], K = num_groups * 128

// 출력  
uint32_t* out_bvr      // [K/32, 8] = [num_groups * 4, 8]
float* scales          // [num_groups]
```

### rtn_7_sbvr_1xtN_mm_T

```cpp
// 입력 (left - 양자화된 활성화)
uint32_t* l_bvr        // [K/32, 8] - RTN 비트 벡터
float* l_scales        // [K/128] - 그룹별 스케일

// 입력 (right - SBVR 양자화된 가중치)
uint32_t* r_bvr        // [K, N, r_num_sums] - SBVR 비트 벡터
uint8_t* r_coeff_idx   // [K, N] - 계수 인덱스 (cache_size <= 256이면 uint8, 아니면 uint16)
__half* r_coeff_cache  // [cache_size, r_num_sums] - 계수 룩업 테이블

// 출력
__half* out            // [1, N]
__half* bias           // [N] (선택적)
```

## 문제 해결

### "sbvr package not found" 오류

```bash
cd /home/wjbang/workspace/sbvr
pip install -e .
```

### "ncu not found" 오류

CUDA Toolkit이 설치되어 있고 PATH에 추가되어 있는지 확인:

```bash
# ncu 위치 찾기
which ncu

# 또는 CUDA 경로 직접 지정
export PATH=/usr/local/cuda/bin:$PATH
```

### NCU 권한 오류

전체 메트릭 수집을 위해 sudo 사용:

```bash
sudo ./run_ncu_profile.sh --kernel rtn_sbvr_mm
```

또는 `/etc/modprobe.d/`에 적절한 설정 추가 (영구적 해결).

## 예제 워크플로우

### 전체 분석 워크플로우

```bash
# 1. 먼저 빠른 벤치마크로 성능 확인
python eval_sbvr_kernels.py --mode benchmark --kernel all --config llama_7b_ffn_up

# 2. 관심 있는 커널 NCU 프로파일링
sudo ./run_ncu_profile.sh --kernel rtn_sbvr_mm --config llama_7b_ffn_up --roofline

# 3. GUI에서 분석
ncu-ui ./ncu_reports/rtn_sbvr_mm_llama_7b_ffn_up_nRTN7.ncu-rep
```

### 다양한 nRTN 비교

```bash
# 모든 nRTN 값으로 벤치마크
python eval_sbvr_kernels.py --mode benchmark --kernel rtn_sbvr_mm --nRTN 7 5 3 --config medium --save-results nrtn_comparison.json
```
