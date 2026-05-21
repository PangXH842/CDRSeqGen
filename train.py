import os
import logging
import sys
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EsmTokenizer,
    EvalPrediction,
    set_seed
)

from dataset import AntibodyTokenizer, AntibodyInfillingDataset, collate_fn
from model import get_model, CustomAntibodyConfig, CustomAntibodyModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

# --- 1. Argument Parser Function ---

def setup_args() -> Tuple[any, any, TrainingArguments]:
    """Parses CLI arguments and sets up the timestamped output directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # Check if a JSON file was passed for arguments (useful for reproducibility and logging)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create timestamped workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_args.model_type.split('/')[-1]}_{timestamp}"
    
    # Update paths
    training_args.output_dir = os.path.join("./results", run_name)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.run_name = run_name

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"📁 Workspace initialized at: {training_args.output_dir}")
    
    return model_args, data_args, training_args

# --- 2. Data Preparation Function ---

def get_datasets(model_args, data_args, training_args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, any]:
    """Handles tokenizer selection, dataset loading, and splitting."""
    # Tokenizer Selection
    if "esm2" in model_args.model_type:
        tokenizer = EsmTokenizer.from_pretrained(model_args.model_type)
    else:
        tokenizer = AntibodyTokenizer()
    
    # Dataset Loading
    full_dataset = AntibodyInfillingDataset(
        data_args.data_path, 
        tokenizer, 
        max_len=data_args.max_len
    )
    
    # Split
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=data_args.test_size, 
        random_state=training_args.seed
    )
    
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    eval_ds = torch.utils.data.Subset(full_dataset, val_idx)
    
    logger.info(f"📊 Data loaded: {len(train_ds)} train, {len(eval_ds)} eval.")
    return train_ds, eval_ds, tokenizer

# --- 3. Training/Execution Function ---

def run_training_pipeline(trainer: Trainer, training_args: TrainingArguments):
    """Executes the train and eval loops based on the provided flags."""
    if training_args.do_train:
        logger.info("🚀 Starting Training...")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if training_args.do_eval:
        logger.info("🔍 Starting Evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Calculates sequence recovery rate (accuracy) only on the masked target loop."""
    predictions, labels = eval_pred
    
    # If the model outputs a tuple (logits, hidden_states), extract the logits
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions

    # Take argmax over the vocab dimension (axis=-1) to get predicted token IDs
    preds = np.argmax(logits, axis=-1)

    # Create a boolean mask where labels are NOT -100 (i.e., the target loop only)
    masked_positions = (labels != -100)

    # Check if there are any masked tokens to evaluate (to avoid division by zero)
    total_masked = masked_positions.sum()
    if total_masked == 0:
        return {"accuracy": 0.0}

    # Calculate how many predictions matched the ground-truth target tokens
    correct_predictions = (preds[masked_positions] == labels[masked_positions]).sum()
    
    # Calculate recovery rate / accuracy
    accuracy = correct_predictions / total_masked

    return {
        "accuracy": float(accuracy)
    }

# --- 4. Main Entry Point ---

def main():
    # 1. Setup
    model_args, data_args, training_args = setup_args()
    set_seed(training_args.seed)

    # 2. Data
    train_dataset, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args)
    
    # 3. Model
    vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else tokenizer.vocab_size
    model = get_model(model_args.model_type, vocab_size=vocab_size)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics, # added
    )

    # 5. Execute
    run_training_pipeline(trainer, training_args)

if __name__ == "__main__":
    main()