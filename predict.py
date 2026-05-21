import argparse
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from dataset import AntibodyTokenizer
from model import CustomAntibodyModel

def generate_sequence(model_path, model_type, pt1, pt2, target_length):
    # 1. Load Tokenizer
    if model_type == "scratch":
        tokenizer = AntibodyTokenizer()
        mask_id = tokenizer.mask_id
    else:
        tokenizer = EsmTokenizer.from_pretrained(model_type)
        mask_id = tokenizer.mask_token_id

    print(f"🧬 Loading model from: {model_path}")
    
    # 2. Load the trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "scratch":
        model = CustomAntibodyModel.from_pretrained(model_path)
    else:
        model = EsmForMaskedLM.from_pretrained(model_path)
        
    model.to(device)
    model.eval() 

    # 3. Prepare the Input Sequence
    # We use add_special_tokens=False to prevent <cls>/<sep> tokens from breaking the middle of the sequence
    if model_type == "scratch":
        pt1_ids = tokenizer.encode(pt1)
        pt2_ids = tokenizer.encode(pt2)
    else:
        pt1_ids = tokenizer.encode(pt1, add_special_tokens=False)
        pt2_ids = tokenizer.encode(pt2, add_special_tokens=False)

    mask_ids = [mask_id] * target_length
    input_ids = pt1_ids + mask_ids + pt2_ids
    
    # Convert to PyTorch Tensor and add batch dimension (1, SequenceLength)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 4. Run the Model
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

    # 5. Extract Predictions
    # The mask tokens are located exactly after pt1
    start_mask_idx = len(pt1_ids)
    end_mask_idx = start_mask_idx + target_length

    # Get the predicted token IDs for the masked positions using greedy decoding (argmax)
    predicted_target_ids = torch.argmax(logits[0, start_mask_idx:end_mask_idx, :], dim=-1).tolist()

    # 6. Decode back to Amino Acids
    if model_type == "scratch":
        predicted_target = tokenizer.decode(predicted_target_ids)
    else:
        predicted_target = tokenizer.decode(predicted_target_ids).replace(" ", "")

    # Print results
    print("\n" + "="*60)
    print(f"🔹 Context (Left / pt1):  {pt1}")
    print(f"🔹 Context (Right / pt2): {pt2}")
    print(f"🔹 Target Gap Length:     {target_length}")
    print("-" * 60)
    print(f"✨ Predicted Infill:      {predicted_target}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Loop Infiller for Antibodies")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default="scratch", help="Type of model: 'scratch' for custom or Hugging Face model name (e.g., 'nvidia/esm2_t12_35M_UR50D')")
    
    # Generic part 1 and part 2 arguments
    parser.add_argument("--pt1", type=str, default="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="WGQGTTVTVSS", help="Sequence AFTER the missing loop")
    parser.add_argument("--target_length", type=int, default=17, help="Number of missing amino acids to predict")
    
    args = parser.parse_args()

    generate_sequence(
        model_path=args.model_path,
        model_type=args.model_type,
        pt1=args.pt1,
        pt2=args.pt2,
        target_length=args.target_length
    )