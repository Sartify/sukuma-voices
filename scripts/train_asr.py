"""
Sukuma ASR Training Script
Fine-tunes Whisper Large V3 for Sukuma speech recognition using Unsloth.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import tqdm
import evaluate
import wandb
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)
from unsloth import FastModel

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Model settings
    MODEL_NAME = "unsloth/whisper-large-v3"
    WHISPER_LANGUAGE = "Swahili"   # this is important
    WHISPER_TASK = "transcribe"
    GENERATION_LANGUAGE = "<|sk|>"
    
    # Dataset settings
    DATASET_NAME = "sartifyllc/SUKUMA_VOICE"
    SAMPLING_RATE = 16000
    
    # Training hyperparameters
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 5
    NUM_EPOCHS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    SEED = 3407
    
    # Logging and evaluation
    LOGGING_STEPS = 500
    EVAL_STEPS = 200
    
    # Output settings
    OUTPUT_DIR = "outputs"
    PUSH_TO_HUB = True
    HUB_MODEL_ID = "sartifyllc/Sukuma-STT"
    
    # W&B settings
    WANDB_PROJECT = "Sukuma ASR"


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment():
    """Configure environment variables."""
    os.environ["HF_AUDIO_DECODER"] = "soundfile"
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


def setup_wandb(config: Config):
    """Initialize Weights & Biases logging."""
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    
    wandb.init(project=config.WANDB_PROJECT)
    wandb.define_metric("test_in_distribution_wer", step_metric="custom_step")
    wandb.define_metric("test_out_distribution_wer", step_metric="custom_step")


# =============================================================================
# Model Setup
# =============================================================================

def load_model(config: Config):
    """Load and configure the Whisper model."""
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.MODEL_NAME,
        dtype=torch.float32,
        load_in_4bit=False,
        auto_model=WhisperForConditionalGeneration,
        whisper_language=config.WHISPER_LANGUAGE,
        whisper_task=config.WHISPER_TASK,
        full_finetuning=True,
    )
    
    model = torch.compiler.disable(model)
    model.gradient_checkpointing_disable()
    
    # Configure generation settings
    model.generation_config.language = config.GENERATION_LANGUAGE
    model.generation_config.task = config.WHISPER_TASK
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    
    return model, tokenizer


# =============================================================================
# Data Processing
# =============================================================================

def create_preprocessing_function(tokenizer):
    """Create a function to preprocess audio examples."""
    def preprocess(example):
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        
        features = tokenizer.feature_extractor(
            audio_array, sampling_rate=sampling_rate
        )
        tokenized_text = tokenizer.tokenizer(example["text"])
        
        return {
            "input_features": features.input_features[0],
            "labels": tokenized_text.input_ids,
        }
    return preprocess


def load_datasets(config: Config, hf_token: str = None):
    """Load and prepare training and test datasets."""
    train_dataset = load_dataset(
        config.DATASET_NAME, 
        split="train", 
        token=hf_token
    )
    test_dataset = load_dataset(
        config.DATASET_NAME, 
        split="test", 
        token=hf_token
    )
    test_dataset_in = load_dataset(
        config.DATASET_NAME, 
        split="test_indistribution_synthesis", 
        token=hf_token
    )
    
    # Resample audio to target sampling rate
    train_dataset = train_dataset.cast_column(
        "audio", Audio(sampling_rate=config.SAMPLING_RATE)
    )
    test_dataset = test_dataset.cast_column(
        "audio", Audio(sampling_rate=config.SAMPLING_RATE)
    )
    test_dataset_in = test_dataset_in.cast_column(
        "audio", Audio(sampling_rate=config.SAMPLING_RATE)
    )
    
    return train_dataset, test_dataset, test_dataset_in


def preprocess_datasets(datasets, preprocess_fn, split_names):
    """Apply preprocessing to all datasets."""
    processed = {}
    for dataset, name in zip(datasets, split_names):
        processed[name] = [
            preprocess_fn(example) 
            for example in tqdm.tqdm(dataset, desc=f"Processing {name}")
        ]
    return processed


# =============================================================================
# Metrics and Data Collator
# =============================================================================

def create_compute_metrics(tokenizer):
    """Create a metrics computation function."""
    metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Handle tuple case (logits, ...)
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        # Convert logits to token IDs
        if len(pred_ids.shape) == 3:
            pred_ids = pred_ids.argmax(axis=-1)
        
        # Replace -100 with pad token
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        # Convert to numpy if needed
        if hasattr(pred_ids, "cpu"):
            pred_ids = pred_ids.cpu().numpy()
        if hasattr(label_ids, "cpu"):
            label_ids = label_ids.cpu().numpy()
        
        # Decode predictions and references
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    
    return compute_metrics


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text models with padding."""
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [
            {"input_features": feature["input_features"]} 
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        
        # Pad labels
        label_features = [
            {"input_ids": feature["labels"]} 
            for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        
        # Mask padding tokens in labels
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present at start of all sequences
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch


# =============================================================================
# Training
# =============================================================================

def create_training_args(config: Config) -> Seq2SeqTrainingArguments:
    """Create training arguments."""
    return Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        logging_steps=config.LOGGING_STEPS,
        eval_steps=config.EVAL_STEPS,
        eval_strategy="steps",
        seed=config.SEED,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="wandb",
    )


def print_gpu_stats(phase: str = "current"):
    """Print GPU memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 0, 0
    
    gpu_stats = torch.cuda.get_device_properties(0)
    reserved_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    
    print(f"\n{'='*50}")
    print(f"GPU Stats ({phase})")
    print(f"{'='*50}")
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved memory: {reserved_memory} GB")
    
    return reserved_memory, max_memory


def print_training_summary(trainer_stats, start_memory, max_memory):
    """Print training summary statistics."""
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    training_memory = round(used_memory - start_memory, 3)
    
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    print(f"Peak memory: {used_memory} GB ({used_memory/max_memory*100:.1f}%)")
    print(f"Training memory: {training_memory} GB ({training_memory/max_memory*100:.1f}%)")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training function."""
    config = Config()
    
    # Get tokens from environment variables
    hf_token = os.environ.get("HF_TOKEN")
    
    # Setup
    setup_environment()
    setup_wandb(config)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(config)
    
    # Load and preprocess data
    print("Loading datasets...")
    train_data, test_data, test_data_in = load_datasets(config, hf_token)
    
    print("Preprocessing datasets...")
    preprocess_fn = create_preprocessing_function(tokenizer)
    processed = preprocess_datasets(
        datasets=[train_data, test_data, test_data_in],
        preprocess_fn=preprocess_fn,
        split_names=["train", "test_out", "test_in"]
    )
    
    # Create trainer components
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)
    compute_metrics = create_compute_metrics(tokenizer)
    training_args = create_training_args(config)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=processed["train"],
        eval_dataset={
            "test_out_distribution": processed["test_out"],
            "test_in_distribution": processed["test_in"],
        },
        data_collator=data_collator,
        tokenizer=tokenizer.feature_extractor,
        compute_metrics=compute_metrics,
        args=training_args,
    )
    
    # Print initial GPU stats
    start_memory, max_memory = print_gpu_stats("before training")
    
    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()
    
    # Evaluate
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate(eval_dataset=processed["test_out"])
    print(f"Test WER (out-of-distribution): {eval_results['eval_wer']:.2f}%")
    
    # Print training summary
    print_training_summary(trainer_stats, start_memory, max_memory)
    
    # Push to Hub
    if config.PUSH_TO_HUB and hf_token:
        print(f"\nPushing model to Hub: {config.HUB_MODEL_ID}")
        model.push_to_hub(config.HUB_MODEL_ID, tokenizer, token=hf_token)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
