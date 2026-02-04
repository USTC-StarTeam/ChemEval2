import json
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input file")
    parser.add_argument("--output_file", required=True, help="Path to output file (includes score details)")
    args = parser.parse_args()

    # ==========================
    # 1. Define Categories
    # ==========================
    # Based on the new data format.
    # Update these sets if the actual values in your data differ slightly.
    knowledge_types = {
        "Multiple Choice Question", 
        "Fill-in-the-blank Question", 
        "Short Answer Question", 
        "Calculation Question", 
        "Chart Understanding"
    }
    
    reaction_types = {
        "Reaction Condition Recommendation", 
        "Reaction Product Prediction", 
        "Reaction Substrate Recommendation"
    }

    # ==========================
    # 2. Initialize Statistics Container
    # ==========================
    stats = {
        'all': {'overall': [], 'final': []},
        'knowledge': {'overall': [], 'final': []},
        'reaction': {'overall': [], 'final': []}
    }

    print(f"Processing file: {args.input_file} ...")
    print(f"Results will be saved to: {args.output_file} ...")

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
         open(args.output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Read basic info
            # Attempt to find checkpoints in various common fields
            checkpoints = data.get("checkpoints", data.get("checkpoint_human", []))
            real_tooltrace = data.get("tooltrace", [])
            # eval_res = data.get("tool_evaluation_result", {})
            eval_res = data.get("evaluation_result", {})
            tool_matched_ids = eval_res.get("tool_matched_ids", [])
            if tool_matched_ids is None: tool_matched_ids = []

            q_type = data.get("question_type", "")

            # # Determine current categories for this sample
            current_categories = ['all']
            if q_type in knowledge_types:
                current_categories.append('knowledge')
            elif q_type in reaction_types:
                current_categories.append('reaction')

            # Build checkpoint_id -> score mapping
            checkpoint_score_map = {}
            for cp in checkpoints:
                try:
                    cid = cp.get("id")
                    score = float(cp.get("score", 0))
                    if cid is not None:
                        checkpoint_score_map[cid] = score
                except:
                    continue

            # --- Calculate Metrics ---
            
            # 1. Overall Score
            t_overall, e_overall = 0.0, 0.0

            for pt in real_tooltrace:
                cid = pt.get("checkpoint")
                s = checkpoint_score_map.get(cid, 0.0)
                is_matched = cid in tool_matched_ids

                # Accumulate Overall
                t_overall += s
                if is_matched:
                    e_overall += s
                
            # --- Calculate Rates and Inject into Data ---

            # Overall Rate
            item_overall_rate = 0.0
            if t_overall > 0:
                item_overall_rate = e_overall / t_overall

            # Save calculated metrics into the data object
            data['tool_metrics'] = {
                'score_overall': item_overall_rate,
                'score_final': 0.0
            }
            
            # Write to output file
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            # --- Store in Stats for Summary ---
            for cat in current_categories:
                # Store overall metric (valid only if denominator > 0)
                if t_overall > 0:
                    stats[cat]['overall'].append(item_overall_rate)

    # ==========================
    # 3. Print Summary Table
    # ==========================
    if not stats['all']['overall']:
        print("No valid data found.")
        return

    # Format Header
    header = f"{'Category':<25} | {'N(Total)':<15} | {'Overall Rate':<20}"
    print("\n" + "="*len(header))
    print(header)
    print("-" * len(header))

    def get_metric_str(score_list):
        if not score_list:
            return "0", "N/A"
        avg = np.mean(score_list)
        return str(len(score_list)), f"{avg:.4f} ({avg*100:.1f}%)"

    # Print rows in order
    rows = [("Total", 'all'), ("Knowledge", 'knowledge'), ("Reaction", 'reaction')]
    
    for label, key in rows:
        d = stats[key]
        
        n_ov, str_ov = get_metric_str(d['overall'])
        
        print(f"{label:<25} | {n_ov:<15} | {str_ov:<20}")

    print("="*len(header) + "\n")
    print(f"Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()