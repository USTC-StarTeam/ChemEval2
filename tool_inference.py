import requests
import json
from typing import Dict, Any, List
import time
import os
import openai
import argparse
from tqdm import tqdm

import config
import utils

# ===================== MCP Tools =====================
with open(config.TOOL_DESCRIPTION, "r", encoding="utf-8") as f:
    CHEM_TOOLS: list[dict] = json.load(f)

def call_fastapi_tool(tool_name: str, params: Dict[str, str]) -> Dict[str, Any]:
    url = f"{config.TOOL_BASE_URL}/{tool_name}"
    try:
        response = requests.post(url, params=params, timeout=60)
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json()
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP Error {e.response.status_code}: {e.response.text}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Call failed: {str(e)}"
        }

# ===================== LLM Model =====================
def call_model_api(model_name, api_key, base_url, messages, tools=None):
    openai.api_key = api_key
    openai.base_url = base_url
    try:
        if tools:
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                timeout=240.0,
                extra_body={
                    "tools": tools or [],
                    "tool_choice": "auto"
                }
            )
        else:
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                timeout=240.0
            )
        return response.dict()
    except Exception as e:
        return {"error": str(e)}

def query_model(messages, tools=None) -> Dict:
    adapted_messages = utils.adapt_messages_for_model(messages, config.API_SETTINGS["model_type"])
    return call_model_api(
        model_name=config.API_SETTINGS["model_name"],
        api_key=config.API_SETTINGS["api_key"],
        base_url=config.API_SETTINGS["base_url"],
        messages=adapted_messages,
        tools=tools
    )

def call_local_model_api(model_name, api_key, messages, tools=None):
    openai.api_key = api_key
    openai.base_url = f"http://localhost:{config.VLLM_SETTINGS['port']}/v1"

    try:
        if tools:
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=240.0,
                # extra_body={
                #     "tools": tools or [],
                #     "tool_choice": "auto"
                # }
                extra_body={
                    "tools": tools,
                    "tool_choice": "none"
                }
            )
        else:
            response = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=240.0
            )
        return response.dict()
    except Exception as e:
        return {"error": str(e)}

def query_local_model(messages: List[Dict], tools=None) -> Dict:
    return call_local_model_api(
        model_name=config.VLLM_SETTINGS["model_path"],
        api_key="EMPTY",
        messages=messages,
        tools=tools
    )


def process_questions(json_paths: List[str], save_path: str, template_path: str):
    with open(template_path, "r", encoding="utf-8") as f:
                    SYSTEM_PROMPT = f.read()
    processed_ids = utils.load_processed_ids(save_path)

    for json_path in json_paths:
        abs_json_path = os.path.abspath(json_path)
        if not os.path.exists(abs_json_path):
            print(f"⚠️  File does not exist: {abs_json_path}, skipping")
            continue
        
        print(f"\n=====================================")
        print(f"Starting to process file:{abs_json_path}")
        print(f"=====================================\n")
        
        with open(abs_json_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f) - len(processed_ids)
            line_num = 0
            f.seek(0)
            for line in tqdm(f, desc="Inference", total=total_lines):
                line_num += 1
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
                try:
                    data = json.loads(line_stripped)
                except json.JSONDecodeError as e:
                    continue

                sample_id = data.get("id")
                if sample_id is None:
                    continue

                if sample_id in processed_ids:
                    continue

                if "tool" not in sample_id:
                    continue
                
                question = data.get("question")
                img_paths = data.get("img_path", [])
                if not question:
                    continue

                if not isinstance(img_paths, list):
                    img_paths = [img_paths] if img_paths else []
                img_paths = [p.strip() for p in img_paths if p and p.strip()]
                
                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    utils.build_multimodal_user_message(question, img_paths)
                ]
                
                tool_trace = []
                called_tool_combinations = {}
                repeat_fail_count = {}
                disable_tools = False
                
                while True:
                    try:
                        if disable_tools:
                            messages.append({
                                "role": "system",
                                "content": (
                                    "Tool usage has failed multiple times. "
                                    "Do NOT call any tools. "
                                    "Please directly provide the best possible final answer."
                                )
                            })
                            model_response = query_model(messages, tools=[])     
                        else:
                            model_response = query_model(messages, CHEM_TOOLS)

                        if not model_response.get("choices"):
                            model_response = query_model(messages, tools=[])

                        message = model_response["choices"][0]["message"]
                        tool_calls = message.get("tool_calls", [])

                        if not tool_calls:
                            record = data
                            record["tool_trace"] = tool_trace
                            record["model_prediction"] = message.get("content", "")

                            with open(save_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            break

                        if disable_tools:
                            messages.append({
                                "role": "assistant",
                                "content": message.get("content", "")
                            })
                            continue

                        for tc in tool_calls:
                            tool_name = tc["function"]["name"]
                            args = json.loads(tc["function"]["arguments"])
                            args_hash = utils.get_args_hash(args)
                            combination_key = f"{tool_name}_{args_hash}"

                            trace_index = len(tool_trace)
                            tool_trace.append({"tool": tool_name, "args": args, "result": None})

                            repeat_fail_count[combination_key] = repeat_fail_count.get(combination_key, 0) + 1
                            if repeat_fail_count[combination_key] >= config.MAX_FAIL:
                                disable_tools = True
                                break

                            if combination_key in called_tool_combinations:
                                time.sleep(0.5)
                                continue 

                            result = call_fastapi_tool(tool_name, args)
                            
                            if not result.get("success", False):
                                result = {
                                    "success": False,
                                    "error": f"{tool_name} return failed: {result.get('error') or 'Unknown error'}",
                                }

                            called_tool_combinations[combination_key] = result
                            tool_trace[trace_index]["result"] = result
                            
                            messages.append({
                                "role": "user", 
                                # "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tool_name,
                                "content": json.dumps(result, ensure_ascii=False)
                            })

                    except Exception as e:
                        break

def process_questions_local(json_paths: List[str], save_path: str, template_path: str):
    processed_ids = utils.load_processed_ids(save_path)
    for json_path in json_paths:
        abs_json_path = os.path.abspath(json_path)
        if not os.path.exists(abs_json_path):
            print(f"⚠️  File does not exist: {abs_json_path}, skipping")
            continue
        
        print(f"\n=====================================")
        print(f"Starting to process file:{abs_json_path}")
        print(f"=====================================\n")
        
        with open(abs_json_path, "r", encoding="utf-8") as f:
            line_num = 0
            total_lines = sum(1 for _ in f) - len(processed_ids)
            f.seek(0)
            for line in tqdm(f, desc="Inference", total=total_lines):
                line_num += 1
                line_stripped = line.strip()
                
                if not line_stripped:
                    continue
                
                try:
                    data = json.loads(line_stripped)
                except json.JSONDecodeError as e:
                    continue

                sample_id = data.get("id")
                if sample_id is None:
                    continue

                if sample_id in processed_ids:
                    continue

                if "tool" not in sample_id:
                    continue
                
                question = data.get("question")
                img_paths = data.get("img_path", [])

                if not question:
                    continue

                if not isinstance(img_paths, list):
                    img_paths = [img_paths] if img_paths else []
                img_paths = [p.strip() for p in img_paths if p and p.strip()]

                with open(template_path, "r", encoding="utf-8") as f:
                    template = f.read()
                tools_str = utils.format_tools_for_prompt(CHEM_TOOLS)
                SYSTEM_PROMPT = template.replace("{{TOOLS_HERE}}", tools_str)
                
                messages = [
                    {   
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    utils.build_multimodal_user_message(question, img_paths)
                ]
                
                tool_trace = []
                called_tool_combinations = {}
                repeat_fail_count = {}
                disable_tools = False

                cnt = 0
                
                while True:
                    try:
                        if disable_tools:
                            messages.append({
                                "role": "system",
                                "content": (
                                    "Tool usage has failed multiple times."
                                    "Do NOT call any tools."
                                    "Please directly provide the best possible final answer."
                                )
                            })
                            model_response = query_local_model(messages, tools=[])
                        else:
                            model_response = query_local_model(messages, CHEM_TOOLS)
                        

                        message = model_response["choices"][0]["message"]
                        tool_calls = utils.extract_tool_calls_from_content(message)

                        if not tool_calls:
                            record = data
                            record["tool_trace"] = tool_trace
                            record["model_prediction"] = message.get("content", "")

                            with open(save_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            break

                        if disable_tools:
                            messages.append({
                                "role": "assistant",
                                "content": message.get("content", "")
                            })
                            continue

                        for tc in tool_calls:
                            tool_name = tc["function"]["name"]
                            args = tc["function"]["arguments"] 
                            args_hash = config.get_args_hash(args)
                            combination_key = f"{tool_name}_{args_hash}"

                            trace_index = len(tool_trace)
                            tool_trace.append({"tool": tool_name, "args": args, "result": None})

                            repeat_fail_count[combination_key] = repeat_fail_count.get(combination_key, 0) + 1
                            if repeat_fail_count[combination_key] >= config.MAX_FAI:
                                disable_tools = True
                                break

                            if combination_key in called_tool_combinations:
                                time.sleep(0.5)
                                continue 

                            result = call_fastapi_tool(tool_name, args)
                            
                            if not result.get("success", False):
                                result = {
                                    "success": False,
                                    "error": f"{tool_name} return failed: {result.get('error') or 'Unknown error'}",
                                }

                            called_tool_combinations[combination_key] = result
                            tool_trace[trace_index]["result"] = result

                            cnt += 1
                            messages.append({
                                "role": "assistant",
                                "tool_call_id": str(cnt),
                                "name": tool_name,
                                "content": json.dumps(result, ensure_ascii=False)
                            })

                    except Exception as e:
                        break


def main():
    parser = argparse.ArgumentParser(
        description="Run ChemEval2 question processing"
    )

    parser.add_argument( "--input_file", nargs="+", required=True, help="Input jsonl file(s)")
    parser.add_argument("--output_file", required=True, help="Output jsonl file path")
    parser.add_argument("--template_path", required=True, help="Path to the template file")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    template_path = args.template_path

    if config.TEST_MODE == "local_vllm":
        process_questions_local(input_path, save_path=output_path, template_path=template_path)
    else:
        process_questions(input_path, save_path=output_path, template_path=template_path)

if __name__ == "__main__":
    main()

