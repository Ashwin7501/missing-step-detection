#!/usr/bin/env python3
"""
Demo Script - Shows exact output format for PPT/Report/Viva

Run with: python demo.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import Procedure
from src.step_extraction import StepExtractor
from src.missing_step_detection import MissingStepDetector, DetectionResult


def run_demo():
    """Run demonstration with sample procedures."""

    print("\n" + "=" * 70)
    print("  MISSING STEP DETECTION SYSTEM - DEMONSTRATION")
    print("  Automatic Detection of Implicit Steps in Procedural Text")
    print("=" * 70)

    # Initialize detector with adjusted thresholds for fallback similarity
    # When sentence-transformers embeddings aren't available, use word-overlap based similarity
    # which produces lower scores, so we use appropriate thresholds
    detector = MissingStepDetector(config={
        "semantic_gap_threshold": 0.15,  # Lower threshold for fallback word-overlap similarity
        "min_confidence_threshold": 0.75,  # Higher confidence required to reduce false positives
        "transition_anomaly_threshold": 0.15,
        "essential_step_threshold": 0.8,
    })

    # =========================================================================
    # DEMO 1: Software Installation (from your example)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  DEMO 1: Software Installation Procedure")
    print("#" * 70)

    procedure1 = Procedure(
        id="demo_software",
        title="How to Install Software",
        steps=[
            "Download the installer.",
            "Run the application.",
            "Use the software.",
        ]
    )

    result1 = detector.detect_missing_steps(procedure1)
    print(detector.generate_explanation(result1, use_emoji=False))

    # =========================================================================
    # DEMO 2: Making Tea (missing boiling step)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  DEMO 2: Making Tea Procedure")
    print("#" * 70)

    procedure2 = Procedure(
        id="demo_tea",
        title="How to Make Tea",
        steps=[
            "Fill the kettle with water.",
            "Pour hot water into the cup with tea bag.",
            "Wait for 3-5 minutes.",
            "Remove tea bag and enjoy.",
        ]
    )

    result2 = detector.detect_missing_steps(procedure2)
    print(detector.generate_explanation(result2, use_emoji=False))

    # =========================================================================
    # DEMO 3: Changing a Tire (multiple missing steps)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  DEMO 3: Changing a Tire Procedure")
    print("#" * 70)

    procedure3 = Procedure(
        id="demo_tire",
        title="How to Change a Flat Tire",
        steps=[
            "Pull over to a safe location.",
            "Get the spare tire from trunk.",
            "Raise the vehicle with the jack.",
            "Remove the lug nuts.",
            "Put on the spare tire.",
            "Drive to a repair shop.",
        ]
    )

    result3 = detector.detect_missing_steps(procedure3)
    print(detector.generate_explanation(result3, use_emoji=False))

    # =========================================================================
    # DEMO 4: Complete procedure (no missing steps)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  DEMO 4: Complete Procedure (No Gaps)")
    print("#" * 70)

    procedure4 = Procedure(
        id="demo_complete",
        title="How to Wash Hands",
        steps=[
            "Turn on the water.",
            "Wet your hands under running water.",
            "Apply soap to your hands.",
            "Rub hands together to create lather.",
            "Scrub all surfaces for 20 seconds.",
            "Rinse hands under running water.",
            "Dry hands with a clean towel.",
        ]
    )

    result4 = detector.detect_missing_steps(procedure4)
    print(detector.generate_explanation(result4, use_emoji=False))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  DEMONSTRATION SUMMARY")
    print("=" * 70)
    print(f"""
    Procedures analyzed: 4

    Demo 1 (Software):  {result1.num_gaps_detected} gap(s) detected
    Demo 2 (Tea):       {result2.num_gaps_detected} gap(s) detected
    Demo 3 (Tire):      {result3.num_gaps_detected} gap(s) detected
    Demo 4 (Hands):     {result4.num_gaps_detected} gap(s) detected

    Detection Methods Used:
    - Semantic similarity analysis
    - Action-object coherence checking
    - Transition probability analysis

    Models: BERT embeddings + similarity computation
    """)
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
