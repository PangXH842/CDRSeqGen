import os
import argparse
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    TrainingArguments, 
    Trainer, 
    EsmTokenizer, 
    DataCollatorForLanguageModeling
)

from dataset import AntibodyTokenizer, AntibodyInfillingDataset, collate_fn
from model import get_model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="scratch", choices=["scratch", "facebook/esm2_t6_8M_UR50D"])
    parser.add_argument("--data_path", type=str, default="dataset/data/processed/processed_paired_data.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # 1. Setup Tokenizer
    if args.model_type == "scratch":
        tokenizer = AntibodyTokenizer()
        vocab_size = len(tokenizer.vocab)
    else:
        # If using ESM, we use their official tokenizer
        tokenizer = EsmTokenizer.from_pretrained(args.model_type)
        vocab_size = tokenizer.vocab_size

    # 2. Prepare Datasets (Train/Val Split)
    # Note: For real research, split by CDR3 identity, but 90/10 random is fine for a test run
    full_dataset = AntibodyInfillingDataset(args.data_path, tokenizer)
    train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.1, random_state=42)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # 3. Load Model
    model = get_model(args.model_type, vocab_size=vocab_size)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",  # Run eval after every epoch
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none", # You can change this to "wandb" later
    )

    # 5. Initialize Trainer
    # We use our custom collate_fn from dataset.py
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # 6. Train!
    print("🚀 Starting training...")
    trainer.train()

    # 7. Save the final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"✅ Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    train()