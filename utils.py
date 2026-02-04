import os
import re
from typing import List, Dict
import config
import base64
import mimetypes
import json
import hashlib
from typing import Dict, Any, List

def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""
    if not os.path.exists(image_path):
        print(f"[Warning] Image not found: {image_path}")
        return None
        
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{base64_data}"

def clean_json_string(s):
    """
    Cleans the LLM output string to extract valid JSON.
    Removes Markdown code blocks and extracts the content between the first { and last }.
    """
    if not s:
        return ""
    s = s.strip()
    # Remove markdown code blocks
    s = re.sub(r"^```(json)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s)
    
    # Extract JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1:
        s = s[start : end + 1]
    return s.strip()

def build_multimodal_user_message(question: str, img_paths: List[str]) -> Dict:
    img_paths = [p for p in (img_paths or []) if p]

    if not img_paths:
        return {"role": "user", "content": question}

    abs_img_paths = [os.path.join(config.DATASET_ROOT_DIR, p) for p in img_paths]

    image_slots = question.count("<ImageHere>")

    if image_slots > 0:
        if len(abs_img_paths) == 1 and image_slots > 1:
            abs_img_paths_for_text = abs_img_paths * image_slots
        else:
            abs_img_paths_for_text = abs_img_paths

        if image_slots > len(abs_img_paths_for_text):
            raise ValueError(
                f"<ImageHere> count ({image_slots}) exceeds img_paths ({len(abs_img_paths)})"
            )

        for abs_path in abs_img_paths_for_text[:image_slots]:
            question = question.replace("<ImageHere>", abs_path, 1)

    content = []
    content.append({
        "type": "text",
        "text": question
    })

    for abs_img_path in abs_img_paths:
        if config.IMG_INPUT_FORMAT == "local_path":
            content.append({
                "type": "image",
                "image_path": abs_img_path
            })
        elif config.IMG_INPUT_FORMAT == "url":
            img_url = f"http://localhost:8000/images/{os.path.basename(abs_img_path)}"
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        elif config.IMG_INPUT_FORMAT == "base64":
            img_url = encode_image_to_base64(abs_img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

    return {"role": "user", "content": content}


def load_processed_ids(output_file):
    if not os.path.exists(output_file):
        return set()

    valid_lines = []
    processed_ids = set()
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                pred = data.get("model_prediction", "")
                
                if pred and str(pred).strip():
                    processed_ids.add(data.get("id")) 
                    valid_lines.append(line)
            except json.JSONDecodeError:
                continue

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line + "\n") 
    print(f"✅ Loaded {len(processed_ids)} valid historical records; these will be skipped.")

    return processed_ids

def adapt_messages_for_model(messages: List[Dict], model_type: str) -> List[Dict]:
    adapted_messages = messages.copy()
    
    if model_type.startswith("GPT"):
        has_tool_calls = any(m.get("tool_calls") for m in adapted_messages if m["role"] == "assistant")
        new_messages = []
        tool_content = None
        for msg in adapted_messages:
            if msg["role"] == "tool":
                tool_content = msg["content"]
                continue
            new_messages.append(msg)
        
        if tool_content and not has_tool_calls:
            if new_messages[-1]["role"] == "user":
                new_messages[-1]["content"] += f"\nTool Call Result：{tool_content}"
            else:
                new_messages.append({
                    "role": "user",
                    "content": f"Tool Call Result：{tool_content}"
                })
        adapted_messages = new_messages
    elif model_type.startswith("Claude"):
        new_messages = []
        tool_content = None
        for msg in adapted_messages:
            if msg["role"] == "tool":
                tool_content = msg["content"]
                continue
            new_messages.append(msg)
        
        if tool_content:
            if new_messages[-1]["role"] == "user":
                new_messages[-1]["content"] += f"\nTool Call Result：{tool_content}"
            else:
                new_messages.append({
                    "role": "user",
                    "content": f"Tool Call Result：{tool_content}"
                })
        adapted_messages = new_messages
    
    return adapted_messages

def get_args_hash(args: Dict[str, Any]) -> str:
    sorted_args = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(sorted_args.encode('utf-8')).hexdigest()

def format_tools_for_prompt(tools):
    lines = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            name = fn["name"]
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
        else:
            name = t["name"]
            desc = t.get("description", "")
            params = t.get("parameters", {})

        lines.append(
            f"- Tool name: {name}\n"
            f"  Description: {desc}\n"
            f"  Parameters (JSON Schema): {params}\n"
        )

    return "\n".join(lines)

def extract_tool_calls_from_content(message):
    content = message.get("content", "")
    if not content or not isinstance(content, str):
        return []

    content = content.strip()

    action_blocks = re.findall(
        r"<\|action_start\|>\s*(?:<\|plugin\|>\s*)?(.*?)\s*<\|action_end\|>",
        content,
        flags=re.DOTALL
    )

    for block in action_blocks:
        block = block.strip()
        try:
            obj = json.loads(block)
            tool_call = _normalize_action_call(obj)
            if tool_call:
                return [tool_call]
        except Exception:
            pass

    tool_calls = _try_parse_tool_calls(content)
    if tool_calls:
        return tool_calls

    code_blocks = re.findall(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.DOTALL
    )

    for block in code_blocks:
        block = block.strip()
        tool_calls = _try_parse_tool_calls(block)
        if tool_calls:
            return tool_calls

    return []

def _normalize_action_call(obj):
    if not isinstance(obj, dict):
        return None

    if "name" in obj and "arguments" in obj and isinstance(obj["arguments"], dict):
        return {
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": obj["arguments"]
            }
        }

    return None

def _try_parse_tool_calls(text):
    try:
        obj = json.loads(text)

        bare = _normalize_bare_call(obj)
        if bare:
            return [bare]

        if isinstance(obj, list):
            return [x for x in obj if _is_tool_call(x)]

        if isinstance(obj, dict) and "tool_calls" in obj:
            return [x for x in obj["tool_calls"] if _is_tool_call(x)]

        if isinstance(obj, dict) and _is_tool_call(obj):
            return [obj]

    except Exception:
        pass

    return []

def _normalize_bare_call(obj):
    if (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and isinstance(obj.get("arguments"), dict)
    ):
        return {
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": obj["arguments"]
            }
        }
    return None

def _is_tool_call(obj):
    if not isinstance(obj, dict):
        return False

    if "function" in obj and isinstance(obj["function"], dict):
        return "name" in obj["function"]

    # {"function": "predict_reaction", "arguments": {...}}
    if isinstance(obj.get("function"), str):
        return True

    return False