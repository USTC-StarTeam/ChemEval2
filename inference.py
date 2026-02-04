import json
import argparse
import os
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import config
from utils import encode_image_to_base64

def get_client_and_model():
    if config.TEST_MODE == "local_vllm":
        base_url = f"http://localhost:{config.VLLM_SETTINGS['port']}/v1"
        api_key = "EMPTY"
        model_name = config.VLLM_SETTINGS["served_model_name"]
    else:
        base_url = config.API_SETTINGS["base_url"]
        api_key = config.API_SETTINGS["api_key"]
        model_name = config.API_SETTINGS["model_name"]
    return OpenAI(api_key=api_key, base_url=base_url), model_name

def load_processed_ids(output_file):
    """
    Loads IDs that have been successfully processed.
    Cleans the file by removing lines with empty predictions/failures.
    Returns a set of IDs to skip.
    """
    if not os.path.exists(output_file):
        return set()

    print(f"🔄 Checkpoint file found: {output_file}. Checking for completed items...")
    valid_lines = []
    processed_ids = set()
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                pred = data.get("model_prediction", "")
                
                # Logic: Only mark as 'processed' if prediction is NOT empty
                if pred and str(pred).strip():
                    processed_ids.add(data.get("id"))
                    valid_lines.append(line)
            except json.JSONDecodeError:
                continue

    # Rewrite the file with only valid lines (cleaning up failures)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line + "\n")
            
    print(f"✅ Loaded {len(processed_ids)} valid records. These will be skipped.")
    return processed_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--template_path", required=True)
    args = parser.parse_args()

    client, model_name = get_client_and_model()
    env = Environment(loader=FileSystemLoader(os.path.dirname(args.template_path)))
    template = env.get_template(os.path.basename(args.template_path))
    
    # 1. Load History (Resume Logic)
    processed_ids = load_processed_ids(args.output_file)

    # 2. Read Input Data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"--- Starting Inference | Mode: {config.TEST_MODE} | Model: {model_name} ---")
    print(f"--- Total: {len(lines)} | Pending: {len(lines) - len(processed_ids)} ---")

    # 3. Open in Append Mode
    with open(args.output_file, 'a', encoding='utf-8', buffering=1) as f_out:
        
        for line in tqdm(lines, desc="Inference"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # --- Skip Logic ---
            current_id = data.get("id")
            if current_id in processed_ids:
                continue
            if "tool" in current_id:
                continue
            # ------------------
            
            # Prepare Images
            raw_img_paths = data.get("img_path", [])
            # Ensure it is a list
            if isinstance(raw_img_paths, str): 
                img_paths = [raw_img_paths]
            elif isinstance(raw_img_paths, list): 
                img_paths = raw_img_paths
            else: 
                img_paths = []
            
            has_image = len(img_paths) > 0

            # Render Prompt
            text_prompt = template.render(question=data.get("question", ""), has_image=has_image)
            
            # Construct Payload
            content_payload = [{"type": "text", "text": text_prompt}]
            
            for rel_path in img_paths:
                full_path = os.path.join(config.DATASET_ROOT_DIR, rel_path)
                b64 = encode_image_to_base64(full_path)
                if b64:
                    content_payload.append({"type": "image_url", "image_url": {"url": b64}})

            # Call API
            prediction = ""
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": content_payload}],
                    temperature=config.INFERENCE_PARAMS["temperature"],
                    max_tokens=config.INFERENCE_PARAMS["max_tokens"],
                    timeout=config.INFERENCE_PARAMS["timeout"]
                )
                prediction = resp.choices[0].message.content
            except Exception as e:
                print(f"[Error] ID {current_id}: {e}")
                prediction = "" # Keep empty so it retries next time
            
            # Save Data
            data['model_input_prompt'] = text_prompt
            data['model_prediction'] = prediction
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
