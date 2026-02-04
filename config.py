import os
import openai
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# ================= Path Configuration =================
# Root directory
DATASET_ROOT_DIR = BASE_DIR / "dataset"

# ==============================================================================
#  "local_vllm"
#  "remote_api"
# ==============================================================================
TEST_MODE = "remote_api"

# ===================== Image Set =====================
IMG_INPUT_FORMAT = "base64"
MAX_FAIL = 3

# ===================== MCP Tool=====================
TOOL_BASE_URL = "http://localhost:9797"
TOOL_DESCRIPTION = BASE_DIR / "chemtool" / "chem_mcp" / "tools_description.json"

# ===================== API-based Model =====================
API_SETTINGS = {
    "model_name": "gpt-5.2-2025-12-11",        
    "api_key": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx",  # Replace with Judge Key     
    "base_url": "https://chatapi.onechats.top/v1/",
    "model_type": "GPT" # (GPT / Qwen / DeepSeek / Gemini / Claude)
}

# ===================== Local Model =====================
VLLM_SETTINGS = {
    "model_path": "/xxxx/xxxx/Qwen3-8B",  
    "served_model_name": "test-model",                
    "port": 8787,                                      
    "gpu_memory_utilization": 0.8,                     
    "max_model_len": 8192,                             
    "trust_remote_code": True,                        
}

# ================= Inference Parameters =================
INFERENCE_PARAMS = {
    "temperature": 0.5,
    "max_tokens": 4096,
    "timeout": 300
}

# ================= Judge Model Settings =================
JUDGE_SETTINGS = {
    "model_name": "gpt-4o",
    "api_key": "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx",  # Replace with Judge Key
    "base_url": "https://api.openai.com/v1",
    "timeout": 120
}