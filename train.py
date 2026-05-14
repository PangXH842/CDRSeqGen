import os
import logging
import sys
import datetime
from dataclasses import dataclass, field
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EsmTokenizer,
    set_seed
)

# Import your custom modules
from dataset import AntibodyTokenizer, AntibodyInfillingDataset, collate_fn
from model import get_model

# --- 1. Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 2. Define Argument Groups ---

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_type: str = field(
        default="scratch",
        metadata={"help": "Path to pretrained model or 'scratch' for our custom transformer."}
    )
    hidden_size: int = field(
        default=128,
        metadata={"help": "Hidden size for the scratch model."}
    )
    num_layers: int = field(
        default=4,
        metadata={"help": "Number of transformer layers for the scratch model."}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input."""
    data_path: str = field(
        default="data/processed/processed_paired_data.json",
        metadata={"help": "The path to the processed JSON data file."}
    )
    max_len: int = field(
        default=128,
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    test_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of data to use for validation."}
    )

# --- 3. Main Training Logic ---

def main():
    # Parse arguments from CLI
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    # If using a json file for args, use: parser.parse_json_file(json_file="args.json")
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create a timestamp string (e.g., 20260514_153022)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define the run name (helps identify what was running)
    run_name = f"{model_args.model_type.split('/')[-1]}_{timestamp}"
    
    # Update the training arguments output_dir
    # If the user provided "./output", it becomes "./results/20260514_153022/"
    base_results_dir = "./results"
    training_args.output_dir = os.path.join(base_results_dir, run_name)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    
    # Update the run_name for W&B logging if enabled
    training_args.run_name = run_name

    # Create the directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    logger.info(f"All results will be saved to: {training_args.output_dir}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    logger.info(f"Training Model: {model_args.model_type}")
    logger.info(f"Data Path: {data_args.data_path}")

    # 4. Tokenizer Selection
    if "esm2" in model_args.model_type:
        tokenizer = EsmTokenizer.from_pretrained(model_args.model_type)
    else:
        tokenizer = AntibodyTokenizer()
    
    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size

    # 5. Dataset Preparation
    full_dataset = AntibodyInfillingDataset(
        data_args.data_path, 
        tokenizer, 
        max_len=data_args.max_len
    )
    
    # Split indices
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=data_args.test_size, 
        random_state=training_args.seed
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    eval_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # 6. Initialize Model
    # We pass hidden_size and layers in case we are using 'scratch'
    model = get_model(model_args.model_type, vocab_size=vocab_size)

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # 8. Run Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    # 9. Run Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()