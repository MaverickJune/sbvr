#!/bin/bash
# ============================================================================
# SBVR Kernel Profiling Script using Nsight Compute (NCU)
# ============================================================================
# This script provides convenient wrappers for profiling SBVR CUDA kernels
# with Nsight Compute.
#
# Requirements:
#   - CUDA Toolkit with ncu (Nsight Compute CLI)
#   - sbvr package installed (pip install -e .)
#   - Root/sudo access (for full profiling capabilities)
#
# Usage:
#   ./run_ncu_profile.sh [options]
#
# Examples:
#   ./run_ncu_profile.sh --kernel rtn_sbvr_mm --config medium
#   ./run_ncu_profile.sh --kernel all --quick
#   sudo ./run_ncu_profile.sh --kernel fused_rtn_lut_bvr --full-metrics
# ============================================================================

set -e

# sudo 실행시 cuda 경로 설정
export PATH="/usr/local/cuda-12.4/bin:$PATH"

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/ncu_reports"
KERNEL="all"
CONFIG="llama_70b_ffn_gate" # 기본값을 llama_70b_ffn_gate로 설정
NRTN=7
METRICS_SET="full"
PYTHON=/home/wjbang/anaconda3/envs/sbvr_nc/bin/python # 항상 conda 환경의 python 사용

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -k, --kernel KERNEL     Kernel to profile (fused_rtn_lut_bvr, rtn_sbvr_mm, fused_rtn_sbvr_mm, or all)"
    echo "  -c, --config CONFIG     Test configuration (small, medium, large, llama_7b_attn, llama_7b_ffn_up, llama_7b_ffn_down, llama_70b_ffn_gate)"
    echo "  -n, --nRTN N            RTN bit width (3, 5, or 7, default: 7)"
    echo "  -o, --output DIR        Output directory for NCU reports"
    echo "  -m, --metrics SET       Metrics set (full, roofline, summary)"
    echo "  --quick                 Quick profiling (summary metrics only)"
    echo "  --full-metrics          Full detailed metrics collection"
    echo "  --roofline              Roofline analysis"
    echo "  --benchmark             Run benchmark mode (no NCU, just timing)"
    echo "  --list-configs          List available configurations"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Profile rtn_sbvr_mm kernel with medium config"
    echo "  $0 --kernel rtn_sbvr_mm --config medium"
    echo ""
    echo "  # Quick profiling of all kernels"
    echo "  $0 --kernel all --quick"
    echo ""
    echo "  # Full roofline analysis"
    echo "  sudo $0 --kernel rtn_sbvr_mm --roofline"
}

# List available configurations
list_configs() {
    echo "Available configurations:"
    echo "  small         - N=2048, K=2048"
    echo "  medium        - N=4096, K=4096"
    echo "  large         - N=8192, K=8192"
    echo "  llama_7b_attn - N=4096, K=4096 (attention projection)"
    echo "  llama_7b_ffn_up   - N=11008, K=4096 (FFN up projection)"
    echo "  llama_7b_ffn_down - N=4096, K=11008 (FFN down projection)"
    echo "  llama_70b_ffn_gate - N=28672, K=8192 (FFN gate projection for 70B model)"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--kernel)
            KERNEL="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -n|--nRTN)
            NRTN="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--metrics)
            METRICS_SET="$2"
            shift 2
            ;;
        --quick)
            METRICS_SET="summary"
            shift
            ;;
        --full-metrics)
            METRICS_SET="full"
            shift
            ;;
        --roofline)
            METRICS_SET="roofline"
            shift
            ;;
        --benchmark)
            MODE="benchmark"
            shift
            ;;
        --list-configs)
            list_configs
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo -e "${RED}Error: ncu (Nsight Compute CLI) not found${NC}"
    echo "Please install CUDA Toolkit or add ncu to your PATH"
    exit 1
fi

# Check if sbvr is installed
if ! $PYTHON -c "import sbvr" &> /dev/null; then
    echo -e "${RED}Error: sbvr package not found${NC}"
    echo "Please install sbvr: cd ${SCRIPT_DIR}/.. && pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get configuration dimensions
get_config_dims() {
    local cfg=$1
    case $cfg in
        small) echo "2048 2048" ;;
        medium) echo "4096 4096" ;;
        large) echo "8192 8192" ;;
        llama_7b_attn) echo "4096 4096" ;;
        llama_7b_ffn_up) echo "11008 4096" ;;
        llama_7b_ffn_down) echo "4096 11008" ;;
        llama_70b_ffn_gate) echo "28672 8192" ;;
        *) echo "4096 4096" ;;
    esac
}

# Get kernel filter pattern for NCU
get_kernel_filter() {
    local kernel=$1
    local nrtn=$2
    case $kernel in
        fused_rtn_lut_bvr) echo "fused_rtn${nrtn}_lut_bvr" ;;
        rtn_sbvr_mm) echo "rtn_${nrtn}_sbvr_1xtN_mm_T" ;;
        fused_rtn_sbvr_mm) echo "rtn_${nrtn}_sbvr_1xtN_mm_T|fused_rtn${nrtn}_lut_bvr" ;;
        *) echo "$kernel" ;;
    esac
}

# Run benchmark mode
if [[ "$MODE" == "benchmark" ]]; then
    echo -e "${BLUE}Running benchmark mode...${NC}"
    $PYTHON "${SCRIPT_DIR}/eval_sbvr_kernels.py" \
        --mode benchmark \
        --kernel "$KERNEL" \
        --config "$CONFIG" \
        --nRTN $NRTN
    exit 0
fi

# Run NCU profiling
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SBVR Kernel NCU Profiling${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Kernel:     ${YELLOW}$KERNEL${NC}"
echo -e "Config:     ${YELLOW}$CONFIG${NC}"
echo -e "nRTN:       ${YELLOW}$NRTN${NC}"
echo -e "Metrics:    ${YELLOW}$METRICS_SET${NC}"
echo -e "Output:     ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Get dimensions
read N K <<< $(get_config_dims "$CONFIG")

# Handle 'all' kernels
if [[ "$KERNEL" == "all" ]]; then
    KERNELS=("fused_rtn_lut_bvr" "rtn_sbvr_mm" "fused_rtn_sbvr_mm")
else
    KERNELS=("$KERNEL")
fi

# Profile each kernel
for kernel in "${KERNELS[@]}"; do
    echo -e "${BLUE}Profiling: $kernel${NC}"
    
    filter=$(get_kernel_filter "$kernel" "$NRTN")
    report_name="${kernel}_${CONFIG}_nRTN${NRTN}"
    report_path="${OUTPUT_DIR}/${report_name}.ncu-rep"
    
    # Build NCU command
    NCU_CMD="ncu"
    NCU_CMD+=" --set $METRICS_SET"
    NCU_CMD+=" --kernel-name-base function"
    NCU_CMD+=" --kernel-name regex:$filter"
    NCU_CMD+=" -o $report_path"
    NCU_CMD+=" --force-overwrite"
    NCU_CMD+=" --target-processes all"
    NCU_CMD+=" $PYTHON ${SCRIPT_DIR}/kernel_runner.py"
    NCU_CMD+=" --kernel $kernel"
    NCU_CMD+=" --N $N --K $K"
    NCU_CMD+=" --nRTN $NRTN"
    
    echo "Command: $NCU_CMD"
    echo ""
    
    # Run NCU
    if eval $NCU_CMD; then
        echo -e "${GREEN}✓ Profile saved: $report_path${NC}"
    else
        echo -e "${RED}✗ Profiling failed for $kernel${NC}"
    fi
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Profiling Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To view reports:"
echo "  ncu-ui ${OUTPUT_DIR}/<report>.ncu-rep"
echo ""
echo "To export to CSV:"
echo "  ncu --import ${OUTPUT_DIR}/<report>.ncu-rep --csv > analysis.csv"
