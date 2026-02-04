#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ========================================================
# 1. Configuration (Edit this section)
# ========================================================

# --- GPU Configuration ---
# Which GPUs to use (comma-separated IDs)
GPUS="0,1"
# Tensor Parallel Size (Must match the number of GPUs provided above)
TP_SIZE=2

# --- File Paths ---
INPUT_FILE="dataset/annotations/chemeval2.0_dataset.jsonl" 
OUTPUT_DIR="examples/experiments/run_$(date +%Y%m%d_%H%M%S)"
VLLM_LOG="$OUTPUT_DIR/vllm_server.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "   Configuration Loaded:"
echo "   GPUS: $GPUS"
echo "   TP_SIZE: $TP_SIZE"
echo "   INPUT_FILE: $INPUT_FILE"
echo "   OUTPUT_DIR: $OUTPUT_DIR"

# ========================================================
# 2. Read Python Config
# ========================================================
# Use python -c to read variables from config.py into Shell variables
TEST_MODE=$(python -c "import config; print(config.TEST_MODE)")

# ========================================================
# Helper Function: Cleanup Background Processes
# ========================================================
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo ""
        echo "Shutting down vLLM service (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        wait $VLLM_PID 2>/dev/null || true
        echo "Service cleaned up."
    fi
}
trap cleanup EXIT INT TERM

# ========================================================
# Step 0: Start vLLM if in Local Mode
# ========================================================

if [ "$TEST_MODE" == "local_vllm" ]; then
    echo "---------------------------------------"
    echo "Local mode detected, preparing to start vLLM..."
    
    # Read vLLM parameters from config.py
    MODEL_PATH=$(python -c "import config; print(config.VLLM_SETTINGS['model_path'])")
    PORT=$(python -c "import config; print(config.VLLM_SETTINGS['port'])")
    SERVED_NAME=$(python -c "import config; print(config.VLLM_SETTINGS['served_model_name'])")
    GPU_UTIL=$(python -c "import config; print(config.VLLM_SETTINGS['gpu_memory_utilization'])")
    MAX_LEN=$(python -c "import config; print(config.VLLM_SETTINGS['max_model_len'])")
    
    echo "   Model Path: $MODEL_PATH"
    echo "   Port: $PORT"
    echo "   Log File: $VLLM_LOG"
    
    # Start command using the variables defined at the top
    CUDA_VISIBLE_DEVICES=$GPUS python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_NAME" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_LEN" \
        --tensor-parallel-size $TP_SIZE \
        --trust-remote-code \
        --enforce-eager \
        > "$VLLM_LOG" 2>&1 &
    
    VLLM_PID=$!
    
    # Wait for Readiness
    echo "Waiting for service to start (PID: $VLLM_PID)..."
    
    # Use curl to check if status is 200
    while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/models)" != "200" ]; do
        # Check if process is still alive
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM startup failed (Process exited). Check log:"
            cat "$VLLM_LOG"
            exit 1
        fi
        sleep 5
        echo -n "."
    done
    echo ""
    echo "vLLM service is ready!"

else
    echo "---------------------------------------"
    echo "API mode detected, skipping local deployment."
fi

# ========================================================
# Step 1: Inference
# ========================================================
echo ""
echo "Step 1: Starting Inference..."
python inference.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/inference.jsonl" \
    --template_path "templates/inference_prompt.jinja2"

if [ "$TEST_MODE" = "local_vllm" ]; then
    python tool_inference.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_DIR/tool_inference.jsonl" \
        --template_path "templates/local_model_inference.prompt"
else
    python tool_inference.py \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_DIR/tool_inference.jsonl" \
        --template_path "templates/model_inference.prompt"
fi

# ========================================================
# Step 2: Evaluation
# ========================================================
echo ""
echo "Step 2: Starting Evaluation..."
python evaluate.py \
    --input_file "$OUTPUT_DIR/inference.jsonl" \
    --output_file "$OUTPUT_DIR/evaluated.jsonl" \
    --template_path "templates/judge_prompt.jinja2"
python evaluate.py \
    --input_file "$OUTPUT_DIR/tool_inference.jsonl" \
    --output_file "$OUTPUT_DIR/tool_evaluated.jsonl" \
    --template_path "templates/judge_prompt.jinja2"
python tool_evaluate.py \
    --input_file "$OUTPUT_DIR/tool_evaluated.jsonl" \
    --output_file "$OUTPUT_DIR/tooltrace_evaluated.jsonl" \
    --template_path "templates/tool_judge_prompt.jinja2"

# # ========================================================
# # Step 3: Metrics Calculation
# # ========================================================
echo ""
echo "Step 3: Calculating Metrics..."
# This step now calculates metrics and modifies the jsonl file
python metrics.py \
    --input_file "$OUTPUT_DIR/tooltrace_evaluated.jsonl" \
    --output_file "$OUTPUT_DIR/final_metrics.jsonl"

python metrics.py \
    --input_file "$OUTPUT_DIR/tooltrace_evaluated.jsonl" \
    --output_file "$OUTPUT_DIR/tool_final_metrics.jsonl"
python tool_metrics.py \
    --input_file "$OUTPUT_DIR/tool_final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/tooltrace_final_metrics.jsonl"

# # ========================================================
# # Step 4: Extract Clean Report
# # ========================================================
echo ""
echo "Step 4: Extracting Final Clean Report..."
python extract_results.py \
    --input_file "$OUTPUT_DIR/final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/final_clean_report.jsonl"
python tool_extract_results.py \
    --input_file "$OUTPUT_DIR/tooltrace_final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/tool_final_clean_report.jsonl"

echo ""
echo "Task Completed! Results directory: $OUTPUT_DIR"
# Cleanup trap will automatically close vLLM upon exit