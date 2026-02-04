# ChemEval2 Inference & Evaluation Pipeline

本项目提供了一套完整的自动化流程，用于对科学领域多模态模型进行推理（Inference）、基于裁判模型的评估（Evaluation）以及最终的指标统计（Metrics）。此外，还包含数据清洗工具，用于生成精简的评测报告。

支持 **本地 vLLM 部署** 和 **远程 API 调用** 两种运行模式。

## 1. 环境准备 (Environment Setup)

在开始之前，请确保 Python 环境（建议 3.9+）已经就绪，并根据硬件情况配置 CUDA 环境（如需本地部署）。

### 安装依赖
请确保项目根目录下包含 `requirements.txt` 和 `tool_requirements.txt`文件，分别为主程序和MCP工具集的依赖并通过以下命令安装所需依赖库：

```bash
pip install -r requirements.txt
pip install -r tool_requirements.txt
```

---

## 2. 配置文件 (Configuration)

运行项目前，必须在根目录下创建并配置 `config.py` 文件。该文件控制整个 pipeline 的运行逻辑。

**需要配置的主要模块包括：**

1. **全局模式选择 (`TEST_MODE`)**：
* 设置为 `local_vllm`：使用本地显卡启动模型服务。
* 设置为 `remote_api`：调用外部模型 API（如 GPT-4, Claude 等）。


2. **路径设置**：
* 指定输入数据的图片根目录。
* 指定本地模型的权重路径 (Checkpoint Path)。


3. **API 密钥**：
* 配置推理模型（如使用远程模式）的 API Key。
* 配置裁判模型（Judge Model，通常为 GPT-4）的 API Key。


4. **硬件参数**：
* 针对本地模式，配置显存占用比例 (`gpu_memory_utilization`)。
* 配置并行卡数 (Tensor Parallelism)。

---

## 3. 运行教程 (Execution Guide)

调用工具需要开启chemtool/chem_mcp中的MCP服务：**uvicorn chem_mcp.app:app --host 0.0.0.0 --port 9797**

本项目提供两种运行方式：**Shell 脚本一键自动化** 和 **Python 分步执行**。

### 方式 A：Shell 脚本一键运行（推荐）

使用 `run_pipeline.sh` 脚本可以自动化管理整个生命周期，包括服务启动、推理、评测和统计。

**步骤：**

1. **修改 GPU 配置**：
打开 `run_pipeline.sh`，根据您的机器实际情况修改显卡编号（如 `CUDA_VISIBLE_DEVICES`）。
2. **执行脚本**：
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```



**脚本主要行为：**

* **服务自启**：若配置为本地模式，脚本会自动在后台启动 vLLM 服务并等待端口就绪。
* **全流程执行**：依次串行执行 推理 -> 评测 -> 统计。
* **自动清理**：任务完成或中断后，脚本会自动杀掉后台的模型服务进程，释放显存。

---

### 方式 B：Python 分步手动执行

如果您需要调试特定环节，或通过其他方式管理模型服务，可按以下顺序分步操作。

#### Step 0: 启动模型服务 (仅 Local 模式)

*如果 config.py 中设置为 `remote_api` 模式，请跳过此步。*

在一个单独的终端窗口中启动 vLLM 服务。请确保端口号 (`--port`) 与 `config.py` 中的设置一致。

```bash
# 示例：使用两张显卡启动 Qwen 模型
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "/path/to/your/model_weights" \
    --served-model-name "test-model" \
    --port 7800 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --tensor-parallel-size 2
```

#### Step 1: 模型推理 (Inference)

运行 `inference.py` 生成模型回答。

* **输入**：原始测试集 `.jsonl` 文件
* **输出**：包含模型预测结果的 `.jsonl` 文件

```bash
python inference.py \
    --input_file "examples/ChemEval2_text.jsonl" \
    --output_file "examples/experiments/inference_output.jsonl" \
    --template_path "templates/inference_prompt.jinja2"
```

运行 `tool_inference.py` 生成工具增强推理答案。

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
```bash

> **提示**：此步骤支持断点续传。如果程序中断，再次运行相同的命令，脚本会自动跳过已生成的 ID。

#### Step 2: 裁判评测 (Evaluation)

运行 `evaluate.py` 调用裁判模型（如 GPT-4）进行打分。

* **输入**：上一步生成的 `inference_output.jsonl`
* **输出**：包含打分详情的 `evaluated_output.jsonl`

```bash
python evaluate.py \
    --input_file "examples/experiments/inference_output.jsonl" \
    --output_file "examples/experiments/evaluated_output.jsonl" \
    --template_path "templates/judge_prompt.jinja2"
```

调用裁判模型（如 GPT-4）对工具调用结果进行打分。

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

#### Step 3: 指标统计 (Metrics)

运行 `metrics.py` 和 `tool_metrics.py` 计算最终得分率。

* **输入**：上一步生成的 `evaluated_output.jsonl`
* **输出**：包含分数的完整数据 `final_metrics.jsonl`

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

#### Step 4: 结果提取 (Extract Results)

运行 `extract_results.py` 和 `tool_extract_results.py` 生成精简版报告（包含ID、预测、裁判理由及分数）。

* **输入**：上一步生成的 `final_metrics.jsonl`
* **输出**：精简后的 `final_clean_report.jsonl`

```bash
python extract_results.py \
    --input_file "examples/experiments/final_metrics.jsonl" \
    --output_file "examples/experiments/final_clean_report.jsonl"
python tool_extract_results.py \
    --input_file "$OUTPUT_DIR/tooltrace_final_metrics.jsonl" \
    --output_file "$OUTPUT_DIR/tool_final_clean_report.jsonl"
```

---

## 4. 产出文件 (Outputs)

流程结束后，输出目录（默认为 `experiments/` 下的带时间戳文件夹）将包含以下关键文件：

| 文件名 | 内容描述 |
| --- | --- |
| **inference.jsonl、tool_inference.jsonl** | 包含模型的原始预测内容 (`model_prediction`、`tool_trace`)。 |
| **evaluated.jsonl、 tool_evaluated.jsonl、 tooltrace_evaluated.jsonl** | 在前者基础上增加了 `evaluation_result`、 `tool_evaluation_result` 字段，包含裁判的打分详情。 |
| **final_metrics.final_metrics.jsonl、 tooltrace_final_metrics.jsonl** | 包含所有字段的完整结果，增加了 `metrics` 对象（`score_overall`, `score_final`, `tool_score_overall`, `tool_score_final`）。 |
| **final_clean_report.jsonl、 tool_final_clean_report.jsonl** | **精简报告**：仅提取 ID、模型回答、裁判理由及最终分数，便于快速查阅和上传。 |