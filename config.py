"""
Configuration settings for the Missing Step Detection System.

This module contains all configurable parameters for the NLP pipeline,
including model hyperparameters, file paths, and analysis settings.
"""

import os
from pathlib import Path

# =============================================================================
# Directory Paths
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_CONFIG = {
    "name": "wikihow",  # Dataset identifier
    "source": "huggingface",  # Source: 'huggingface', 'local', or 'synthetic'
    "hf_dataset_name": "wikihow",  # HuggingFace dataset name
    "hf_subset": "all",  # Dataset subset
    "max_samples": 5000,  # Maximum samples to load (None for all)
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
}

# =============================================================================
# Text Preprocessing Configuration
# =============================================================================
PREPROCESSING_CONFIG = {
    "min_step_length": 5,  # Minimum characters per step
    "max_step_length": 500,  # Maximum characters per step
    "min_steps_per_procedure": 3,  # Minimum steps in a procedure
    "max_steps_per_procedure": 30,  # Maximum steps in a procedure
    "remove_urls": True,
    "remove_special_chars": False,
    "lowercase": False,  # Keep original case for NER
    "remove_stopwords": False,  # Keep stopwords for context
}

# =============================================================================
# NLP Pipeline Configuration
# =============================================================================
NLP_CONFIG = {
    "spacy_model": "en_core_web_sm",  # SpaCy model for NER/POS
    "sentence_transformer": "all-MiniLM-L6-v2",  # For semantic similarity
    "similarity_threshold": 0.7,  # Threshold for step similarity
    "action_verbs_path": None,  # Path to custom action verbs list
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_CONFIG = {
    # Transformer model settings
    "transformer_model": "bert-base-uncased",
    "max_seq_length": 256,
    "hidden_size": 768,
    "num_attention_heads": 12,

    # Sequence model settings
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
    "lstm_bidirectional": True,

    # Training settings
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,

    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_delta": 0.001,
}

# =============================================================================
# Missing Step Detection Configuration
# =============================================================================
DETECTION_CONFIG = {
    # Transition analysis
    "transition_anomaly_threshold": 0.3,  # Below this = potential gap
    "semantic_gap_threshold": 0.4,  # Semantic discontinuity threshold

    # Confidence scoring
    "min_confidence_threshold": 0.5,  # Minimum confidence to report
    "high_confidence_threshold": 0.8,  # High confidence threshold

    # Step classification
    "essential_step_threshold": 0.7,  # Threshold for essential vs optional

    # Inference settings
    "max_inferred_steps": 3,  # Max steps to infer between two steps
    "use_beam_search": True,
    "beam_width": 5,
    "temperature": 0.7,
}

# =============================================================================
# Analysis Configuration
# =============================================================================
ANALYSIS_CONFIG = {
    "n_gram_range": (1, 3),  # For transition analysis
    "top_k_patterns": 20,  # Top patterns to report
    "clustering_n_clusters": 10,  # For step clustering
    "min_pattern_frequency": 5,  # Minimum occurrences for pattern
}

# =============================================================================
# Output Configuration
# =============================================================================
OUTPUT_CONFIG = {
    "save_analysis_plots": True,
    "save_model_checkpoints": True,
    "checkpoint_frequency": 1,  # Save every N epochs
    "log_level": "INFO",
    "results_format": "json",  # 'json' or 'csv'
}

# =============================================================================
# Evaluation Configuration
# =============================================================================
EVAL_CONFIG = {
    "metrics": ["precision", "recall", "f1", "accuracy"],
    "num_qualitative_examples": 10,
    "cross_validation_folds": 5,
}
