import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import sys
import json

# Add the project root to the Python path to allow for absolute imports
# This is a robust way to handle imports in a structured project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets.dataset import InfillingDataset, infilling_collate_fn, vocab
from model.transformers import InfillingTransformer

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Performs one full pass over the training set."""
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training Epoch"):
        # Move batch data to the correct device (GPU/CPU)
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        padding_mask = batch['padding_mask'].to(device)

        # 1. Zero the gradients before running the backward pass.
        optimizer.zero_grad()

        # 2. Forward pass
        logits = model(
            src_tokens=input_tokens,
            padding_mask=padding_mask
        )

        # 3. Calculate loss
        # The model predicts logits for all positions, but we only care about the masked ones.
        # For this baseline, we'll calculate loss on all tokens and let the loss function ignore padding.
        # A more advanced version would create a specific loss mask for only the CDR-H3 region.
        
        # Reshape for CrossEntropyLoss, which expects [N, C] and [N]
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
        
        # 4. Backward pass
        loss.backward()

        # 5. Update the weights
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    """Performs one full pass over the validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Epoch"):
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            logits = model(
                src_tokens=input_tokens,
                padding_mask=padding_mask
            )
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main(args):
    """Main function to orchestrate the training process."""
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # --- Data Loading ---
    print("Loading data...")
    train_dataset = InfillingDataset(data_file=args.data_file, split='train')
    val_dataset = InfillingDataset(data_file=args.data_file, split='validation')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=infilling_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=infilling_collate_fn, num_workers=4)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")

    # --- Model Initialization ---
    print("Initializing model...")
    model = InfillingTransformer(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created with ~{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters")

    # --- Loss and Optimizer ---
    # `ignore_index` is crucial for telling the loss function to ignore padding tokens.
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.token_to_idx["<pad>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_val_loss = float('inf')
    print("\n--- Starting Training ---")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'args': args # Save the config as well
            }, checkpoint_path)
            print(f"Validation loss improved. Saved checkpoint to {checkpoint_path}")

    print("\n--- Training Finished ---")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer for CDR-H3 Infilling")
    
    # --- Arguments ---
    parser.add_argument('--data_file', type=str, default='datasets/data/processed/sabdab_processed_H_chain_infill.pt', help='Path to the processed data file.')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save training results.')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    
    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimension of embeddings.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Dimension of the feed-forward network.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    
    args = parser.parse_args()
    
    main(args)