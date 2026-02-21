#!/usr/bin/env python3
"""
Missing Step Detection System - Main Entry Point

This system automatically detects and infers missing/implicit steps
in instructional or procedural text using NLP and deep learning.

Usage:
    python main.py --mode demo          # Run demonstration
    python main.py --mode analyze       # Run analysis only
    python main.py --mode detect        # Run detection pipeline
    python main.py --mode train         # Train models
    python main.py --mode evaluate      # Evaluate performance
    python main.py --input FILE         # Process specific file

Author: NLP System
Dataset: WikiHow / Synthetic Procedural Text
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATASET_CONFIG,
    PREPROCESSING_CONFIG,
    NLP_CONFIG,
    MODEL_CONFIG,
    DETECTION_CONFIG,
    ANALYSIS_CONFIG,
    OUTPUT_CONFIG,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo():
    """Run the demonstration mode."""
    print("\n" + "=" * 70)
    print("MISSING STEP DETECTION SYSTEM")
    print("Automatic Detection and Inference of Missing Steps in Procedural Text")
    print("=" * 70)

    from src.inference import InferencePipeline, PipelineConfig

    config = PipelineConfig(
        use_synthetic_data=True,
        max_samples=50,
        run_analysis=True,
        save_analysis=True,
        save_results=True,
        generate_report=True,
        verbose=True,
        output_dir=str(PROJECT_ROOT / "outputs"),
    )

    pipeline = InferencePipeline(
        config=config,
        preprocessing_config=PREPROCESSING_CONFIG,
        analysis_config=ANALYSIS_CONFIG,
        nlp_config=NLP_CONFIG,
        detection_config=DETECTION_CONFIG,
        model_config=MODEL_CONFIG,
    )

    # Run pipeline
    output = pipeline.run(data_source="synthetic")

    # Interactive demo
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO")
    print("=" * 70)

    from src.data_preprocessing import Procedure

    demo_procedures = [
        Procedure(
            id="demo_cooking",
            title="How to Boil an Egg",
            steps=[
                "Place eggs in a pot.",
                # Missing: Fill pot with water to cover eggs
                # Missing: Place pot on stove
                "Bring to a boil.",
                "Turn off heat and cover.",
                # Missing: Let sit for desired time (6-12 minutes)
                "Transfer eggs to ice bath.",
                "Peel and enjoy.",
            ]
        ),
        Procedure(
            id="demo_tech",
            title="How to Reset Your Password",
            steps=[
                "Go to the login page.",
                "Click 'Forgot Password'.",
                # Missing: Enter your email address
                # Missing: Click submit
                "Check your email for the reset link.",
                # Missing: Click the reset link
                "Enter your new password.",
                # Missing: Confirm your new password
                "Log in with your new password.",
            ]
        ),
    ]

    for proc in demo_procedures:
        print(f"\n--- Analyzing: {proc.title} ---\n")
        result = pipeline.run_single(proc)
        explanation = pipeline.explain_detection(result)
        print(explanation)

    print("\n" + "=" * 70)
    print(f"Results saved to: {config.output_dir}")
    print("=" * 70)

    return output


def run_analysis():
    """Run exploratory analysis on the dataset."""
    print("\n" + "=" * 70)
    print("EXPLORATORY AND STRUCTURAL ANALYSIS")
    print("=" * 70)

    from src.data_preprocessing import DataPreprocessor
    from src.analysis import ProceduralAnalyzer

    # Load data
    preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
    procedures = preprocessor.load_dataset(
        source=DATASET_CONFIG.get("source", "synthetic"),
        max_samples=DATASET_CONFIG.get("max_samples", 100),
    )

    print(f"\nLoaded {len(procedures)} procedures")
    print(f"Dataset statistics: {preprocessor.stats}")

    # Run analysis
    analyzer = ProceduralAnalyzer(ANALYSIS_CONFIG)
    results = analyzer.analyze(procedures)

    # Generate and save report
    report = analyzer.generate_report(
        output_path=str(PROJECT_ROOT / "outputs" / "analysis_report.txt")
    )
    print(report)

    # Save results
    analyzer.save_results(str(PROJECT_ROOT / "outputs" / "analysis_results.json"))

    # Generate plots
    try:
        analyzer.plot_analysis(str(PROJECT_ROOT / "outputs" / "plots"))
        print("\nPlots saved to outputs/plots/")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

    # Print transition table
    print("\n" + analyzer.generate_transition_table())

    return results


def run_detection(input_file: str = None, max_samples: int = None):
    """Run the detection pipeline."""
    print("\n" + "=" * 70)
    print("MISSING STEP DETECTION PIPELINE")
    print("=" * 70)

    from src.inference import InferencePipeline, PipelineConfig
    from src.data_preprocessing import DataPreprocessor, Procedure

    # Determine data source
    procedures = None
    if input_file:
        # Load from file
        input_path = Path(input_file)
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                procedures = [Procedure.from_dict(p) for p in data]
            else:
                procedures = [Procedure.from_dict(data)]
        elif input_path.suffix == '.txt':
            # Assume text file with steps separated by newlines
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            title = input_path.stem
            steps = [line.strip() for line in lines if line.strip()]
            procedures = [Procedure(id="input", title=title, steps=steps)]
        print(f"Loaded {len(procedures)} procedures from {input_file}")

    config = PipelineConfig(
        use_synthetic_data=(procedures is None),
        max_samples=max_samples or DATASET_CONFIG.get("max_samples", 100),
        run_analysis=True,
        save_results=True,
        generate_report=True,
        verbose=True,
        output_dir=str(PROJECT_ROOT / "outputs"),
    )

    pipeline = InferencePipeline(
        config=config,
        preprocessing_config=PREPROCESSING_CONFIG,
        analysis_config=ANALYSIS_CONFIG,
        nlp_config=NLP_CONFIG,
        detection_config=DETECTION_CONFIG,
    )

    output = pipeline.run(procedures=procedures, data_source="synthetic")

    return output


def run_train():
    """Train the sequence models."""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch is required for training. Please install it first.")
        return None

    from src.data_preprocessing import DataPreprocessor, create_negative_samples
    from src.sequence_model import (
        StepCoherenceModel,
        TransitionDataset,
        ModelTrainer,
    )

    # Load and prepare data
    print("\n[1] Loading and preparing data...")
    preprocessor = DataPreprocessor(PREPROCESSING_CONFIG)
    procedures = preprocessor.load_dataset(
        source="synthetic",
        max_samples=MODEL_CONFIG.get("max_samples", 500),
    )

    train_procs, val_procs, test_procs = preprocessor.create_splits()

    # Create positive pairs (consecutive steps)
    print("\n[2] Creating training pairs...")
    positive_pairs = []
    for proc in train_procs:
        for i in range(len(proc.steps) - 1):
            positive_pairs.append((proc.steps[i], proc.steps[i + 1]))

    # Create negative pairs
    negative_samples = create_negative_samples(train_procs, n_negatives_per_positive=1)
    negative_pairs = [(s["current_step"], s["next_step"]) for s in negative_samples]

    print(f"  Positive pairs: {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")

    # Create model and tokenizer
    print("\n[3] Initializing model...")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG.get("transformer_model", "bert-base-uncased"))
    except ImportError:
        print("transformers library required for training. Please install it.")
        return None

    model = StepCoherenceModel(MODEL_CONFIG)

    # Create datasets
    train_dataset = TransitionDataset(
        positive_pairs[:int(len(positive_pairs) * 0.9)],
        negative_pairs[:int(len(negative_pairs) * 0.9)],
        tokenizer,
        max_length=MODEL_CONFIG.get("max_seq_length", 128),
    )

    val_positive = positive_pairs[int(len(positive_pairs) * 0.9):]
    val_negative = negative_pairs[int(len(negative_pairs) * 0.9):]
    val_dataset = TransitionDataset(val_positive, val_negative, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG.get("batch_size", 16),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG.get("batch_size", 16),
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Train
    print("\n[4] Training model...")
    trainer = ModelTrainer(model, MODEL_CONFIG)
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=MODEL_CONFIG.get("num_epochs", 5),
        save_dir=str(PROJECT_ROOT / "models"),
    )

    print("\n[5] Training complete!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"  Model saved to: {PROJECT_ROOT / 'models'}")

    return history


def run_evaluate(model_path: str = None):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    from src.data_preprocessing import DataPreprocessor, Procedure
    from src.inference import InferencePipeline, PipelineConfig
    from src.missing_step_detection import GapEvaluator

    # Create test data with known gaps
    print("\n[1] Creating test data with known gaps...")

    # Test procedures with deliberately removed steps
    test_cases = [
        {
            "title": "How to Make Coffee",
            "complete_steps": [
                "Fill kettle with water.",
                "Boil the water.",
                "Add coffee grounds to filter.",
                "Pour hot water over grounds.",
                "Wait for coffee to brew.",
                "Pour into cup.",
                "Add cream and sugar if desired.",
            ],
            "incomplete_steps": [
                "Fill kettle with water.",
                # Removed: "Boil the water."
                "Add coffee grounds to filter.",
                "Pour hot water over grounds.",
                # Removed: "Wait for coffee to brew."
                "Pour into cup.",
                "Add cream and sugar if desired.",
            ],
            "true_gaps": [0, 3],  # Positions after which steps are missing
        },
        {
            "title": "How to Send an Email",
            "complete_steps": [
                "Open email application.",
                "Click compose.",
                "Enter recipient address.",
                "Write subject line.",
                "Write message body.",
                "Attach files if needed.",
                "Review message.",
                "Click send.",
            ],
            "incomplete_steps": [
                "Open email application.",
                "Click compose.",
                # Removed: "Enter recipient address."
                "Write subject line.",
                "Write message body.",
                # Removed: "Attach files if needed."
                # Removed: "Review message."
                "Click send.",
            ],
            "true_gaps": [1, 3, 3],  # Multiple gaps
        },
    ]

    # Create procedures
    test_procedures = []
    ground_truth = {}

    for i, case in enumerate(test_cases):
        proc = Procedure(
            id=f"test_{i}",
            title=case["title"],
            steps=case["incomplete_steps"],
        )
        test_procedures.append(proc)
        ground_truth[proc.id] = case["true_gaps"]

    # Run detection
    print("\n[2] Running detection on test cases...")

    config = PipelineConfig(
        run_analysis=False,
        save_results=True,
        generate_report=False,
        verbose=False,
        output_dir=str(PROJECT_ROOT / "outputs" / "evaluation"),
    )

    pipeline = InferencePipeline(
        config=config,
        preprocessing_config=PREPROCESSING_CONFIG,
        detection_config=DETECTION_CONFIG,
    )

    output = pipeline.run(
        procedures=test_procedures,
        ground_truth=ground_truth,
    )

    # Print results
    print("\n[3] Evaluation Results:")
    print("-" * 40)

    if output.evaluation_metrics:
        print(f"  Precision: {output.evaluation_metrics['precision']:.3f}")
        print(f"  Recall: {output.evaluation_metrics['recall']:.3f}")
        print(f"  F1 Score: {output.evaluation_metrics['f1']:.3f}")
        print(f"  True Positives: {output.evaluation_metrics['true_positives']}")
        print(f"  False Positives: {output.evaluation_metrics['false_positives']}")
        print(f"  False Negatives: {output.evaluation_metrics['false_negatives']}")

    # Qualitative analysis
    print("\n[4] Qualitative Analysis:")
    print("-" * 40)

    for result, case in zip(output.detection_results, test_cases):
        print(f"\nProcedure: {result.procedure_title}")
        print(f"  True gap positions: {case['true_gaps']}")
        print(f"  Detected positions: {[g.position for g in result.detected_gaps]}")

        for gap in result.detected_gaps:
            print(f"\n  Gap at position {gap.position}:")
            print(f"    Type: {gap.gap_type.value}")
            print(f"    Confidence: {gap.confidence:.2%}")
            if gap.inferred_step:
                print(f"    Inferred: {gap.inferred_step}")

    # Error analysis
    print("\n[5] Error Analysis:")
    print("-" * 40)

    for result, case in zip(output.detection_results, test_cases):
        true_set = set(case["true_gaps"])
        pred_set = set([g.position for g in result.detected_gaps])

        false_positives = pred_set - true_set
        false_negatives = true_set - pred_set

        if false_positives:
            print(f"\n  {result.procedure_title} - False Positives:")
            for pos in false_positives:
                gap = next((g for g in result.detected_gaps if g.position == pos), None)
                if gap:
                    print(f"    Position {pos}: {gap.explanation}")

        if false_negatives:
            print(f"\n  {result.procedure_title} - False Negatives:")
            for pos in false_negatives:
                print(f"    Position {pos}: Gap not detected")

    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Missing Step Detection System for Procedural Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                    Run demonstration
  python main.py --mode analyze                 Run exploratory analysis
  python main.py --mode detect --max-samples 50 Detect missing steps
  python main.py --mode train                   Train models
  python main.py --mode evaluate                Evaluate performance
  python main.py --input steps.txt              Process input file
        """
    )

    parser.add_argument(
        "--mode",
        choices=["demo", "analyze", "detect", "train", "evaluate"],
        default="demo",
        help="Operation mode (default: demo)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (JSON or TXT with steps)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        OUTPUT_CONFIG["output_dir"] = args.output_dir

    # Run selected mode
    if args.mode == "demo":
        run_demo()

    elif args.mode == "analyze":
        run_analysis()

    elif args.mode == "detect":
        run_detection(
            input_file=args.input,
            max_samples=args.max_samples,
        )

    elif args.mode == "train":
        run_train()

    elif args.mode == "evaluate":
        run_evaluate(model_path=args.model_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
