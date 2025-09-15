import torch
import argparse
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets.dataset import vocab 
from model.transformers import InfillingTransformer

def generate_sequence(model, tokenizer, masked_sequence, device):
    """
    Generates a completed sequence by filling in the masked tokens.

    Args:
        model (nn.Module): The trained InfillingTransformer model.
        tokenizer (Vocabulary): The vocabulary object.
        masked_sequence (str): The heavy chain sequence with 'X' as mask tokens.
        device (torch.device): The device to run the model on (cuda or cpu).

    Returns:
        str: The completed antibody sequence.
    """
    model.eval() # Set the model to evaluation mode

    # 1. Prepare the input
    input_tokens = tokenizer.tokenize(masked_sequence)
    # Add a batch dimension (batch size of 1) and move to the correct device
    input_tensor = input_tokens.unsqueeze(0).to(device)

    # The padding mask is simple for a single sequence: all are valid tokens.
    padding_mask = torch.zeros_like(input_tensor, dtype=torch.bool).to(device)

    # 2. Run inference
    with torch.no_grad(): # Disable gradient calculation for efficiency
        logits = model(
            src_tokens=input_tensor,
            padding_mask=padding_mask
        )
        
    # Manually prevent the model from predicting special tokens.
    # We set the logits for <pad> and X to negative infinity before the argmax.
    special_tokens_indices = [tokenizer.token_to_idx['<pad>'], tokenizer.token_to_idx['X']]
    logits[:, :, special_tokens_indices] = -float('inf')

    # 3. Decode the output
    # Get the token with the highest probability at each position
    predicted_indices = torch.argmax(logits, dim=-1)
    # Remove the batch dimension
    predicted_indices = predicted_indices.squeeze(0)

    # 4. Fill in the blanks
    # Start with the original input tokens
    output_tokens = input_tokens.clone()
    
    # Find the positions that were masked ('X')
    mask_positions = (input_tokens == tokenizer.token_to_idx['X'])
    
    # Replace the 'X' tokens with the model's predictions at those positions
    output_tokens[mask_positions] = predicted_indices[mask_positions]

    # 5. Convert token indices back to an amino acid string
    final_sequence = "".join([tokenizer.idx_to_token[idx.item()] for idx in output_tokens])
    
    return final_sequence


def main(args):
    """Main function to load the model and run generation."""
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Load the arguments that the model was trained with
    train_args = checkpoint['args']

    # --- Initialize Model ---
    print("Initializing model architecture...")
    model = InfillingTransformer(
        vocab_size=len(vocab),
        embed_dim=train_args.embed_dim,
        num_layers=train_args.num_layers,
        num_heads=train_args.num_heads,
        hidden_dim=train_args.hidden_dim,
        dropout=train_args.dropout
    ).to(device)
    
    # --- Load Trained Weights ---
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")
    
    # --- Generate Sequence ---
    print("\n--- Generating Sequence ---")
    print(f"Input Masked Sequence: {args.sequence}")
    
    generated_seq = generate_sequence(
        model=model,
        tokenizer=vocab,
        masked_sequence=args.sequence,
        device=device
    )
    
    print(f"\nModel Output Sequence: {generated_seq}")
    print("---------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CDR-H3 sequences with a trained model.")
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--sequence', type=str, required=True, help="The masked heavy chain sequence. Use 'X' for masked positions.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run inference on.')
    
    args = parser.parse_args()
    
    main(args)