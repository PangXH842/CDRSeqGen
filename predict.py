import argparse
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from dataset import AntibodyTokenizer
from model import CustomAntibodyModel

def generate_h3(model_path, model_type, fw1_3, fw4, h3_length):
    # 1. Load the correct Tokenizer
    if model_type == "scratch":
        tokenizer = AntibodyTokenizer()
        mask_id = tokenizer.mask_id
    else:
        # Load the ESM tokenizer
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
    model.eval() # Set to evaluation mode (turns off dropout)

    # 3. Prepare the Input Sequence
    if model_type == "scratch":
        fw1_3_ids = tokenizer.encode(fw1_3)
        fw4_ids = tokenizer.encode(fw4)
    else:
        # ESM tokenizer adds special BOS/EOS tokens automatically if we use __call__,
        # but we want exact control over the sequence.
        fw1_3_ids = tokenizer.encode(fw1_3, add_special_tokens=False)
        fw4_ids = tokenizer.encode(fw4, add_special_tokens=False)

    # Create the masked middle section
    mask_ids = [mask_id] * h3_length
    
    # Combine: FW1-3 + [MASK]*N + FW4
    input_ids = fw1_3_ids + mask_ids + fw4_ids
    
    # Convert to PyTorch Tensor and add batch dimension (1, SequenceLength)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 4. Run the Model
    with torch.no_grad():
        outputs = model(input_ids=input_tensor)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

    # 5. Extract Predictions
    # The mask tokens are located between FW1_3 and FW4
    start_mask_idx = len(fw1_3_ids)
    end_mask_idx = start_mask_idx + h3_length

    # Get the predicted token IDs for the masked positions using greedy decoding (argmax)
    # Logits shape: (Batch=1, SeqLen, VocabSize)
    predicted_h3_ids = torch.argmax(logits[0, start_mask_idx:end_mask_idx, :], dim=-1).tolist()

    # 6. Decode back to Amino Acids
    if model_type == "scratch":
        predicted_h3 = tokenizer.decode(predicted_h3_ids)
    else:
        predicted_h3 = tokenizer.decode(predicted_h3_ids).replace(" ", "")

    # Print results
    print("\n" + "="*50)
    print(f"🔹 Context (FW1-3): {fw1_3}")
    print(f"🔹 Context (FW4):   {fw4}")
    print(f"🔹 Mask Length:     {h3_length}")
    print("-" * 50)
    print(f"✨ Predicted H3:    {predicted_h3}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./results/final_model", help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default="scratch", choices=["scratch", "facebook/esm2_t6_8M_UR50D"])
    
    # Default sequences taken from Jaffe_2022 (Donor 4)
    parser.add_argument("--fw1_3", type=str, default="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR", help="Frameworks 1 through 3 + CDR1 & 2")
    parser.add_argument("--fw4", type=str, default="WGQGTTVTVSS", help="Framework 4")
    parser.add_argument("--h3_length", type=int, default=17, help="Length of the CDR-H3 to generate")
    
    args = parser.parse_args()

    generate_h3(
        model_path=args.model_path,
        model_type=args.model_type,
        fw1_3=args.fw1_3,
        fw4=args.fw4,
        h3_length=args.h3_length
    )