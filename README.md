# ChemEval2 Inference & Evaluation Pipeline

This project provides a comprehensive automated pipeline designed for **inference** of multimodal models in scientific domains, **judge-model-based evaluation**, and final **metrics calculation**. It also includes data cleaning tools for generating streamlined evaluation reports.

The pipeline supports two operating modes: **Local vLLM Deployment** and **Remote API Calls**.

## 1. Environment Setup

Before starting, ensure your Python environment (recommended 3.9+) is ready and configure the CUDA environment based on your hardware (if local deployment is required).

### Install Dependencies

Ensure the project root directory contains `requirements.txt` and `tool_requirements.txt` (dependencies for the main program and MCP toolset respectively), then install them using the following commands:

```bash
pip install -r requirements.txt
pip install -r tool_requirements.txt
```

---

## 2. Configuration

Before running the project, you must create and configure a `config.py` file in the root directory. This file controls the execution logic of the entire pipeline.

**Main modules to configure include:**

1. **Global Mode Selection (`TEST_MODE`)**:
   * Set to `local_vllm`: Uses local GPUs to start model services.
   * Set to `remote_api`: Calls external model APIs (e.g., GPT-4, Claude, etc.).


2. **Path Settings**:
   * Specify the root directory for input images.
   * Specify the local model weight path (Checkpoint Path).


3. **API Keys**:
   * Configure the API Key for the inference model (if using remote mode).
   * Configure the API Key for the Judge Model (usually GPT-4).


4. **Hardware Parameters**:
   * For local mode, configure the GPU memory usage ratio (`gpu_memory_utilization`).
   * Configure Tensor Parallelism (number of parallel GPUs).



---

## 3. Execution Guide

To invoke the tools, you need to start the MCP service in `chemtool/chem_mcp`:

**`uvicorn chem_mcp.app:app --host 0.0.0.0 --port 9797`**

This project provides two execution methods: **One-click Shell Script Automation** and **Step-by-step Python Execution**.

### Method A: One-click Shell Script (Recommended)

Use the `run_pipeline.sh` script to automatically manage the entire lifecycle, including service startup, inference, evaluation, and statistics.

**Steps:**

1. **Modify GPU Configuration**:
Open `run_pipeline.sh` and modify the GPU indices (e.g., `CUDA_VISIBLE_DEVICES`) according to your machine's hardware.
2. **Execute the Script**:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```



**Main Script Actions:**

* **Service Self-start**: If configured for local mode, the script automatically starts the vLLM service in the background and waits for the port to become ready.
* **Full Flow Execution**: Executes Inference -> Evaluation -> Metrics in sequence.
* **Auto Cleanup**: Upon task completion or interruption, the script automatically kills background model service processes to release GPU memory.

---

### Method B: Step-by-step Manual Python Execution

If you need to debug specific stages or manage model services via other means, follow these steps in order.

#### Step 0: Start Model Service (Local Mode Only)

*Skip this step if `config.py` is set to `remote_api` mode.*

Start the vLLM service in a separate terminal window. Ensure the port number (`--port`) matches the setting in `config.py`.

```bash
# Example: Starting a Qwen model with two GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "/path/to/your/model_weights" \
    --served-model-name "test-model" \
    --port 7800 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --tensor-parallel-size 2
```

#### Step 1: Model Inference

Run `inference.py` to generate model answers.

* **Input**: Original test set `.jsonl` file.
* **Output**: `.jsonl` file containing model prediction results.

```bash
python inference.py \
    --input_file "examples/ChemEval2_text.jsonl" \
    --output_file "examples/experiments/inference_output.jsonl" \
    --template_path "templates/inference_prompt.jinja2"
```

Run `tool_inference.py` to generate tool-augmented inference answers.

```bash
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
```

> **Tip**: This step supports checkpointing. If the program is interrupted, re-running the same command will automatically skip IDs that have already been generated.

#### Step 2: Judge Evaluation

Run `evaluate.py` to call a judge model (e.g., GPT-4) for scoring.

* **Input**: The `inference_output.jsonl` generated in the previous step.
* **Output**: `evaluated_output.jsonl` containing scoring details.

```bash
python evaluate.py \
    --input_file "examples/experiments/inference_output.jsonl" \
    --output_file "examples/experiments/evaluated_output.jsonl" \
    --template_path "templates/judge_prompt.jinja2"

```

Call the judge model (e.g., GPT-4) to score tool-calling results.

```bash
python evaluate.py \
    --input_file "$OUTPUT_DIR/tool_inference.jsonl" \
    --output_file "$OUTPUT_DIR/tool_evaluated.jsonl" \
    --template_path "templates/judge_prompt.jinja2"

python tool_evaluate.py \
    --input_file "$OUTPUT_DIR/tool_evaluated.jsonl" \
    --output_file "$OUTPUT_DIR/tooltrace_evaluated.jsonl" \
    --template_path "templates/tool_judge_prompt.jinja2"
```

#### Step 3: Metrics Calculation

Run `metrics.py` and `tool_metrics.py` to calculate the final score rates.

* **Input**: The `evaluated_output.jsonl` generated in the previous step.
* **Output**: Complete data file `final_metrics.jsonl` including scores.

```bash
python metrics.py \
    --input_file "examples/experiments/evaluated_output.jsonl" \
    --output_file "examples/experiments/final_metrics.jsonl"

python metrics.py \
    --input_file "$OUTPUT_DIR/tooltrace_evaluated.jsonl" \
    --output_file "$OUTPUT_DIR/tool_final_metrics.jsonl"

python tool_metrics.py \
    --input_file "$OUTPUT_DIR/tool_final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/tooltrace_final_metrics.jsonl"
```

#### Step 4: Result Extraction

Run `extract_results.py` and `tool_extract_results.py` to generate streamlined reports (containing ID, prediction, judge reasoning, and scores).

* **Input**: The `final_metrics.jsonl` generated in the previous step.
* **Output**: The cleaned report `final_clean_report.jsonl`.

```bash
python extract_results.py \
    --input_file "examples/experiments/final_metrics.jsonl" \
    --output_file "examples/experiments/final_clean_report.jsonl"

python tool_extract_results.py \
    --input_file "$OUTPUT_DIR/tooltrace_final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/tool_final_clean_report.jsonl"
```

---

## 4. Outputs

After the pipeline finishes, the output directory (defaulting to a timestamped folder under `experiments/`) will contain the following key files:

| Filename | Description |
| --- | --- |
| **inference.jsonl, tool_inference.jsonl** | Contains raw model predictions (`model_prediction`, `tool_trace`). |
| **evaluated.jsonl, tool_evaluated.jsonl, tooltrace_evaluated.jsonl** | Builds on the above by adding `evaluation_result` and `tool_evaluation_result` fields with judge scoring details. |
| **final_metrics.jsonl, tooltrace_final_metrics.jsonl** | Full results containing all fields plus a `metrics` object (`score_overall`, `score_final`, `tool_score_overall`, `tool_score_final`). |
| **final_clean_report.jsonl, tool_final_clean_report.jsonl** | **Streamlined Report**: Extracts only IDs, model answers, judge reasoning, and final scores for quick review and upload. |
