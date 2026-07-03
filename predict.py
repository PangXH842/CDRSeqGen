import argparse
import json
import torch
from backbones import resolve

def generate_sequence(model, tokenizer, device, name, pt1, pt2, target_length, target_seq=None):
    """Generates the infill sequence and prints the alignment/results.

    `tokenizer` is a TokenizerAdapter, so this builds inputs identically for any backbone:
    the wrapping ([cls] ... [end] vs none) and the set of banned special-token ids both come
    from the adapter rather than being hardcoded per model family.
    """
    # 1. Prepare the Input Sequence — mirrors AntibodyInfillingDataset.__getitem__ construction.
    pt1_ids = tokenizer.encode(pt1)
    pt2_ids = tokenizer.encode(pt2)
    mask_ids = [tokenizer.mask_id] * target_length

    if tokenizer.uses_special_tokens:
        input_ids = [tokenizer.cls_id] + pt1_ids + mask_ids + pt2_ids + [tokenizer.end_id]
        start_mask_idx = 1 + len(pt1_ids)  # 1 for the leading cls token
    else:
        input_ids = pt1_ids + mask_ids + pt2_ids
        start_mask_idx = len(pt1_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 2. Run the Model
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

    # 3. Extract Predictions
    end_mask_idx = start_mask_idx + target_length

    # Ban every non-amino-acid token so only real residues can be predicted.
    mask_logits = logits[0, start_mask_idx:end_mask_idx, :].clone()
    for sid in tokenizer.special_ids:
        mask_logits[:, sid] = -float("inf")

    predicted_target_ids = torch.argmax(mask_logits, dim=-1).tolist()

    # 4. Decode back to Amino Acids
    predicted_target = tokenizer.decode(predicted_target_ids).upper().strip()

    # 5. Print Results
    print("\n" + "="*65)
    print(f"🧪 Test Case:             {name}")
    print(f"🔹 Context (Left / pt1):  {pt1}")
    print(f"🔹 Context (Right / pt2): {pt2}")
    print(f"🔹 Target Gap Length:     {target_length}")
    print("-" * 65)
    print(f"✨ Predicted Infill:      {predicted_target}")
    
    # 6. Evaluate against Ground Truth (if provided)
    if target_seq:
        target_seq = target_seq.upper().strip()
        eval_len = min(len(target_seq), len(predicted_target))
        match_indicators = []
        correct_count = 0
        
        for i in range(eval_len):
            if predicted_target[i] == target_seq[i]:
                match_indicators.append("|")
                correct_count += 1
            else:
                match_indicators.append(".")
                
        if len(predicted_target) != len(target_seq):
            discrepancy = abs(len(predicted_target) - len(target_seq))
            match_indicators.extend(["?"] * discrepancy)
            
        alignment_str = "".join(match_indicators)
        recovery_rate = (correct_count / len(target_seq)) * 100 if len(target_seq) > 0 else 0
        
        print(f"🍀 Ground Truth:          {target_seq}")
        print(f"🎯 Alignment Map:         {alignment_str}")
        print(f"📈 Sequence Recovery:     {recovery_rate:.1f}%")
        
    print("="*65)


def main(args):
    # --- 1. Load Model and Tokenizer Once (via the backbone registry) ---
    print(f"🧬 Loading model from: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resolve(args.model_type)
    tokenizer = backbone.build_tokenizer(args.model_type)
    model = backbone.load_pretrained_model(args.model_path)

    model.to(device)
    model.eval()

    # --- 2. Process Inputs ---
    if args.test_file:
        # Load from JSON
        with open(args.test_file, 'r') as f:
            test_cases = json.load(f)

        for case in test_cases:
            t_seq = case.get("target_seq", None)
            t_len = len(t_seq) if t_seq else case.get("target_length", 10)

            generate_sequence(
                model=model, tokenizer=tokenizer, device=device,
                name=case.get("name", "Unnamed Test"),
                pt1=case["pt1"], pt2=case["pt2"],
                target_length=t_len, target_seq=t_seq
            )
    else:
        # Load from CLI (Single Test)
        t_len = len(args.target_seq) if args.target_seq else args.target_length
        generate_sequence(
            model=model, tokenizer=tokenizer, device=device,
            name="CLI Test Case",
            pt1=args.pt1, pt2=args.pt2,
            target_length=t_len, target_seq=args.target_seq
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Loop Infiller with JSON Batch Support")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default="scratch", help="Type of model: 'scratch' for custom transformer, or specify ESM-2 model name (eg. 'nvidia/esm2_t12_35M_UR50D')")
    
    # Accept a JSON file of test cases (omit to fall back to single CLI test case below)
    parser.add_argument("--test_file", type=str, default=None, help="Path to a JSON file containing test sequences")
    
    # Fallback CLI arguments for quick single tests
    parser.add_argument("--pt1", type=str, default="", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="", help="Sequence AFTER the missing loop")
    parser.add_argument("--target_length", type=int, default=10, help="Number of missing amino acids to predict")
    parser.add_argument("--target_seq", type=str, default=None, help="Optional Ground Truth")
    
    args = parser.parse_args()

    main(args)