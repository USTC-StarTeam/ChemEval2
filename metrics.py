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
            checkpoints = data.get("checkpoints", data.get("checkpoint_human", data.get("metadata", {}).get("checkpoints", [])))
            eval_res = data.get("evaluation_result", {})
            matched_ids = eval_res.get("matched_ids", [])
            if matched_ids is None: matched_ids = []

            q_type = data.get("question_type", "")

            # Determine current categories for this sample
            current_categories = ['all']
            if q_type in knowledge_types:
                current_categories.append('knowledge')
            elif q_type in reaction_types:
                current_categories.append('reaction')

            # --- Calculate Metrics ---
            
            # 1. Overall Score
            t_overall, e_overall = 0.0, 0.0
            # 2. Final Answer Score (Checkpoints where score == 3.0)
            t_final, e_final = 0.0, 0.0

            for pt in checkpoints:
                try:
                    s = float(pt.get("score", 0))
                except: 
                    s = 0.0
                
                is_matched = pt.get("id") in matched_ids
                
                # Accumulate Overall
                t_overall += s
                if is_matched:
                    e_overall += s
                
                # Accumulate Final Answer (Check if score is approx 3.0)
                if abs(s - 3.0) < 1e-9:
                    t_final += s
                    if is_matched:
                        e_final += s

            # --- Calculate Rates and Inject into Data ---
            
            # Overall Rate
            item_overall_rate = 0.0
            if t_overall > 0:
                item_overall_rate = e_overall / t_overall
            
            # Final Rate (None if no 3.0 score points exist)
            item_final_rate = None 
            if t_final > 0:
                item_final_rate = e_final / t_final

            # Save calculated metrics into the data object
            data['metrics'] = {
                'score_overall': item_overall_rate,
                'score_final': item_final_rate
            }
            
            # Write to output file
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

            # --- Store in Stats for Summary ---
            for cat in current_categories:
                # Store overall metric (valid only if denominator > 0)
                if t_overall > 0:
                    stats[cat]['overall'].append(item_overall_rate)
                
                # Store final metric (valid only if denominator > 0)
                if t_final > 0:
                    stats[cat]['final'].append(item_final_rate)

    # ==========================
    # 3. Print Summary Table
    # ==========================
    if not stats['all']['overall']:
        print("No valid data found.")
        return

    # Format Header
    header = f"{'Category':<35} | {'N(Total)':<8} | {'Overall Rate':<20} | {'N(Final)':<8} | {'Final Rate'}"
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
        n_fi, str_fi = get_metric_str(d['final'])
        
        print(f"{label:<35} | {n_ov:<8} | {str_ov:<20} | {n_fi:<8} | {str_fi}")

    print("="*len(header) + "\n")
    print(f"Detailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()