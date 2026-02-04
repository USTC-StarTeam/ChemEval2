import json
import argparse
import os
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
import config
from utils import encode_image_to_base64, clean_json_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--template_path", required=True)
    args = parser.parse_args()

    client = OpenAI(api_key=config.JUDGE_SETTINGS["api_key"], base_url=config.JUDGE_SETTINGS["base_url"])
    env = Environment(loader=FileSystemLoader(os.path.dirname(args.template_path)))
    template = env.get_template(os.path.basename(args.template_path))

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in.readlines(), desc="Judging"):
            data = json.loads(line)

            # Check if prediction exists (skip empty inferences)
            if not data.get("model_tooltrace"):
                data['tool_evaluation_result'] = {"tool_matched_ids": [], "tool_matched_ids": "No prediction generated"}
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue
            
            model_pretooltrace = data.get("model_tooltrace", [])

            # Render Prompt
            prompt = template.render(
                question=data.get("question", ""),
                real_tooltrace=data.get("tooltrace", []),
                model_tooltrace=model_pretooltrace
            )
            
            # Construct Payload
            content_payload = [{"type": "text", "text": prompt}]

            # Call Judge
            try:
                res = client.chat.completions.create(
                    model=config.JUDGE_SETTINGS["model_name"],
                    messages=[
                        {"role": "system", "content": "You are a strict grader. Output ONLY valid JSON."},
                        {"role": "user", "content": content_payload}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    timeout=config.JUDGE_SETTINGS["timeout"]
                )
                
                raw_json = res.choices[0].message.content
                eval_json = json.loads(clean_json_string(raw_json))
                
            except Exception as e:
                print(f"[Eval Error] ID {data.get('id', '?')}: {e}")
                eval_json = {"tool_matched_ids": [], "tool_reasoning": f"Error: {str(e)}"}

            data['tool_evaluation_result'] = eval_json
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

