"""
Missing Step Detection Module - Core logic for detecting and inferring missing steps.
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np

from .data_preprocessing import Procedure
from .step_extraction import StepExtractor, StructuredStep
from .analysis import TransitionPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GapType(Enum):
    SEMANTIC_DISCONTINUITY = "semantic_discontinuity"
    MISSING_PREREQUISITE = "missing_prerequisite"
    MISSING_INTERMEDIATE = "missing_intermediate"
    ACTION_SEQUENCE_GAP = "action_sequence_gap"
    OBJECT_REFERENCE_GAP = "object_reference_gap"
    TEMPORAL_GAP = "temporal_gap"


class StepImportance(Enum):
    ESSENTIAL = "essential"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass
class DetectedGap:
    position: int
    before_step: str
    after_step: str
    gap_type: GapType
    importance: StepImportance
    confidence: float
    inferred_step: Optional[str] = None
    inferred_action: Optional[str] = None
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    semantic_score: float = 0.0
    transition_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "position": self.position, "before_step": self.before_step, "after_step": self.after_step,
            "gap_type": self.gap_type.value, "importance": self.importance.value,
            "confidence": self.confidence, "inferred_step": self.inferred_step,
            "explanation": self.explanation, "evidence": self.evidence,
        }


@dataclass
class DetectionResult:
    procedure_id: str
    procedure_title: str
    original_steps: List[str]
    detected_gaps: List[DetectedGap]
    reconstructed_steps: List[str]
    num_gaps_detected: int = 0
    num_essential_gaps: int = 0
    avg_confidence: float = 0.0
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "procedure_id": self.procedure_id, "procedure_title": self.procedure_title,
            "original_steps": self.original_steps,
            "detected_gaps": [g.to_dict() for g in self.detected_gaps],
            "reconstructed_steps": self.reconstructed_steps,
            "num_gaps_detected": self.num_gaps_detected,
            "num_essential_gaps": self.num_essential_gaps,
            "avg_confidence": self.avg_confidence,
        }


class MissingStepDetector:
    """Detects and infers missing steps in procedures."""

    INTERMEDIATE_ACTIONS = {
        ('gather', 'mix'): 'measure', ('download', 'use'): 'install',
        ('cut', 'cook'): 'season', ('add', 'serve'): 'cook',
    }

    # Known patterns where a step is commonly missing (action1, action2) -> missing_step
    KNOWN_MISSING_PATTERNS = {
        # Software/Tech
        ("download", "run"): "Install the software.",
        ("download", "use"): "Install and configure the software.",
        ("install", "use"): "Launch the application.",

        # Cooking/Food
        ("fill", "pour"): "Boil the water.",
        ("fill", "add"): "Boil the water.",
        ("cut", "cook"): "Season the ingredients.",
        ("mix", "bake"): "Preheat the oven.",
        ("add", "serve"): "Cook until ready.",
        ("crack", "mix"): "Add to the bowl.",
        ("boil", "drain"): "Cook for the required time.",

        # Cleaning/Hygiene
        ("rinse", "dry"): "Turn off the water.",
        ("wash", "dry"): "Rinse thoroughly.",

        # Vehicle/Mechanical
        ("remove", "put"): "Remove the old component completely.",
        ("loosen", "remove"): "Fully unscrew.",
        ("jack", "remove"): "Loosen the lug nuts first.",
        ("raise", "remove"): "Loosen the lug nuts first.",

        # General
        ("open", "use"): "Prepare the contents.",
        ("gather", "start"): "Read the instructions.",
    }

    # Step transitions that are naturally connected (don't need intermediate steps)
    NATURAL_TRANSITIONS = {
        # Hand washing - these naturally follow each other
        ("turn", "wet"), ("wet", "apply"), ("apply", "rub"), ("rub", "scrub"),
        ("scrub", "rinse"), ("scrub", "continue"),
        # Cooking
        ("stir", "wait"), ("wait", "remove"), ("pour", "wait"), ("pour", "stir"),
        # Vehicle/Mechanical - tire changing
        ("pull", "get"), ("get", "raise"), ("raise", "remove"), ("put", "drive"),
        ("loosen", "raise"), ("tighten", "lower"),
        # Computer/Tech
        ("start", "wait"), ("click", "enter"), ("select", "click"),
        ("download", "install"), ("install", "run"), ("run", "use"),
        ("launch", "use"), ("execute", "use"), ("open", "use"),
        # General
        ("get", "place"), ("place", "raise"), ("check", "start"),
        ("read", "follow"), ("open", "read"),
    }

    def __init__(self, config: Dict = None, step_extractor: StepExtractor = None, analyzer=None):
        self.config = config or {}
        self.step_extractor = step_extractor or StepExtractor(config)
        self.analyzer = analyzer
        self.semantic_gap_threshold = self.config.get("semantic_gap_threshold", 0.4)
        self.transition_anomaly_threshold = self.config.get("transition_anomaly_threshold", 0.3)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.5)
        self.essential_step_threshold = self.config.get("essential_step_threshold", 0.7)

        # Load transition patterns from analyzer
        self.transition_probs = {}
        if analyzer and hasattr(analyzer, 'results') and analyzer.results.transition_patterns:
            for p in analyzer.results.transition_patterns:
                self.transition_probs[(p.from_action, p.to_action)] = p.probability

    def detect_missing_steps(self, procedure: Procedure, methods: List[str] = None) -> DetectionResult:
        start_time = time.time()
        methods = methods or ['semantic', 'transition', 'coherence']

        structured_steps = self.step_extractor.extract_procedure(procedure)
        all_gaps = []

        if 'semantic' in methods:
            all_gaps.extend(self._detect_semantic_gaps(procedure.steps, structured_steps))
        if 'transition' in methods:
            all_gaps.extend(self._detect_transition_gaps(structured_steps))
        if 'coherence' in methods:
            all_gaps.extend(self._detect_coherence_gaps(structured_steps))

        merged = self._merge_gaps(all_gaps)
        filtered = [g for g in merged if g.confidence >= self.min_confidence_threshold]

        for gap in filtered:
            self._infer_step_content(gap, structured_steps)
            self._classify_importance(gap)

        reconstructed = self._reconstruct_procedure(procedure.steps, filtered)
        num_essential = sum(1 for g in filtered if g.importance == StepImportance.ESSENTIAL)
        avg_conf = np.mean([g.confidence for g in filtered]) if filtered else 0.0

        return DetectionResult(
            procedure_id=procedure.id, procedure_title=procedure.title,
            original_steps=procedure.steps, detected_gaps=filtered,
            reconstructed_steps=reconstructed, num_gaps_detected=len(filtered),
            num_essential_gaps=num_essential, avg_confidence=avg_conf,
            processing_time=time.time() - start_time,
        )

    def _detect_semantic_gaps(self, steps, structured_steps):
        gaps = []
        for i in range(len(structured_steps) - 1):
            s1, s2 = structured_steps[i], structured_steps[i+1]
            a1 = s1.main_action.text if s1.main_action else ""
            a2 = s2.main_action.text if s2.main_action else ""

            # Check if this is a known missing pattern (high confidence gap)
            key = (a1, a2)
            if key in self.KNOWN_MISSING_PATTERNS:
                inferred = self.KNOWN_MISSING_PATTERNS[key]
                gaps.append(DetectedGap(
                    position=i, before_step=steps[i], after_step=steps[i+1],
                    gap_type=GapType.MISSING_INTERMEDIATE, importance=StepImportance.ESSENTIAL,
                    confidence=0.9, semantic_score=0.0,
                    inferred_step=inferred,
                    explanation=f"Known missing step pattern: {a1} -> {a2}",
                    evidence=[f"Common procedural gap identified", f"Missing: {inferred}"],
                ))
                continue

            # Check if this is a natural transition (skip it)
            if key in self.NATURAL_TRANSITIONS:
                continue

            # Only flag semantic gaps if similarity is very low AND not a natural transition
            sim = self.step_extractor.compute_similarity(s1, s2)

            # Higher threshold for unknown transitions, but don't flag natural progressions
            if sim < self.semantic_gap_threshold:
                # Check if actions are related (if so, likely not a real gap)
                if self._are_actions_related(a1, a2):
                    continue

                gaps.append(DetectedGap(
                    position=i, before_step=steps[i], after_step=steps[i+1],
                    gap_type=GapType.SEMANTIC_DISCONTINUITY, importance=StepImportance.RECOMMENDED,
                    confidence=max(0.5, 1.0 - sim), semantic_score=sim,
                    explanation=f"Large semantic gap (similarity={sim:.2f})",
                    evidence=[f"Similarity {sim:.2f} < threshold {self.semantic_gap_threshold}"],
                ))
        return gaps

    def _are_actions_related(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically related and naturally follow each other."""
        if not action1 or not action2:
            return False

        related_groups = [
            {'add', 'put', 'place', 'insert', 'pour', 'apply'},
            {'remove', 'take', 'get', 'pull', 'lift'},
            {'mix', 'stir', 'whisk', 'blend', 'combine'},
            {'heat', 'cook', 'boil', 'warm', 'bake'},
            {'cut', 'slice', 'chop', 'dice'},
            {'wash', 'rinse', 'clean', 'wet', 'scrub'},
            {'dry', 'wipe', 'towel'},
            {'open', 'start', 'begin', 'turn'},
            {'close', 'stop', 'end', 'finish'},
            {'click', 'select', 'choose', 'press', 'tap'},
            {'enter', 'type', 'input', 'write'},
            {'download', 'install', 'get', 'save'},
            {'run', 'execute', 'start', 'launch', 'use'},
            {'wait', 'pause', 'hold', 'let'},
            {'check', 'verify', 'ensure', 'confirm', 'inspect'},
            {'rub', 'scrub', 'massage'},
        ]
        for group in related_groups:
            if action1 in group and action2 in group:
                return True
        return False

    def _detect_transition_gaps(self, structured_steps):
        gaps = []
        for i in range(len(structured_steps) - 1):
            s1, s2 = structured_steps[i], structured_steps[i+1]
            a1 = s1.main_action.text if s1.main_action else None
            a2 = s2.main_action.text if s2.main_action else None
            if a1 and a2:
                prob = self.transition_probs.get((a1, a2), 0.0)
                if prob < self.transition_anomaly_threshold:
                    intermediate = self.INTERMEDIATE_ACTIONS.get((a1, a2))
                    gaps.append(DetectedGap(
                        position=i, before_step=s1.original_text, after_step=s2.original_text,
                        gap_type=GapType.ACTION_SEQUENCE_GAP, importance=StepImportance.RECOMMENDED,
                        confidence=0.7 if intermediate else 0.5, transition_score=prob,
                        inferred_action=intermediate,
                        explanation=f"Unusual transition: {a1} -> {a2} (prob={prob:.2f})",
                        evidence=[f"Transition probability {prob:.2f} < {self.transition_anomaly_threshold}"],
                    ))
        return gaps

    def _detect_coherence_gaps(self, structured_steps):
        gaps = []
        introduced_objects = set()
        for i, step in enumerate(structured_steps):
            if step.has_prerequisite and i == 0:
                gaps.append(DetectedGap(
                    position=-1, before_step="[START]", after_step=step.original_text,
                    gap_type=GapType.MISSING_PREREQUISITE, importance=StepImportance.ESSENTIAL,
                    confidence=0.8, explanation="First step has prerequisite indicators",
                    evidence=["No prior steps establish prerequisites"],
                ))
            if step.primary_object:
                obj_words = set(step.primary_object.text.lower().split())
                if not introduced_objects & obj_words and i > 0:
                    gaps.append(DetectedGap(
                        position=i-1, before_step=structured_steps[i-1].original_text if i > 0 else "[START]",
                        after_step=step.original_text, gap_type=GapType.OBJECT_REFERENCE_GAP,
                        importance=StepImportance.RECOMMENDED, confidence=0.6,
                        explanation=f"Object '{step.primary_object.text}' used without introduction",
                        evidence=["Object not mentioned in previous steps"],
                    ))
                introduced_objects.update(obj_words)
        return gaps

    def _merge_gaps(self, gaps):
        if not gaps:
            return []
        gaps.sort(key=lambda g: g.position)
        merged = []
        current = gaps[0]
        for gap in gaps[1:]:
            if gap.position == current.position:
                current.confidence = max(current.confidence, gap.confidence)
                current.evidence.extend(gap.evidence)
                current.inferred_action = current.inferred_action or gap.inferred_action
            else:
                merged.append(current)
                current = gap
        merged.append(current)
        return merged

    def _infer_step_content(self, gap, structured_steps):
        """Infer what missing step content should be based on context."""
        # If we already have an inferred step from known patterns, keep it
        if gap.inferred_step:
            if not gap.inferred_action:
                words = gap.inferred_step.split()
                if words:
                    gap.inferred_action = words[0].lower().rstrip('.')
            return

        # If we have an inferred action but no step
        if gap.inferred_action and not gap.inferred_step:
            if 0 <= gap.position < len(structured_steps):
                obj = structured_steps[gap.position].primary_object
                if obj:
                    gap.inferred_step = f"{gap.inferred_action.capitalize()} the {obj.text}."
                else:
                    gap.inferred_step = f"{gap.inferred_action.capitalize()} as needed."
            return

        # Handle missing prerequisite
        if gap.gap_type == GapType.MISSING_PREREQUISITE:
            gap.inferred_step = "Gather all required materials and tools."
            gap.inferred_action = "gather"
            return

        # Infer based on surrounding steps using common patterns
        if 0 <= gap.position < len(structured_steps) - 1:
            before = structured_steps[gap.position]
            after = structured_steps[gap.position + 1]

            before_action = before.main_action.text if before.main_action else ""
            after_action = after.main_action.text if after.main_action else ""

            # Check known missing patterns
            key = (before_action, after_action)
            if key in self.KNOWN_MISSING_PATTERNS:
                gap.inferred_step = self.KNOWN_MISSING_PATTERNS[key]
                gap.inferred_action = gap.inferred_step.split()[0].lower()
                return

            # Generic inference based on gap type
            if gap.gap_type == GapType.SEMANTIC_DISCONTINUITY:
                # Try to create a bridging step
                if after.primary_object:
                    gap.inferred_step = f"Prepare the {after.primary_object.text}."
                    gap.inferred_action = "prepare"
                elif before.primary_object:
                    gap.inferred_step = f"Complete the {before_action} process."
                    gap.inferred_action = "complete"

            elif gap.gap_type == GapType.OBJECT_REFERENCE_GAP:
                if after.primary_object:
                    gap.inferred_step = f"Get or prepare the {after.primary_object.text}."
                    gap.inferred_action = "prepare"

            elif gap.gap_type == GapType.ACTION_SEQUENCE_GAP:
                gap.inferred_step = f"Complete the intermediate step before {after_action}."
                gap.inferred_action = "complete"

    def _classify_importance(self, gap):
        if gap.gap_type == GapType.MISSING_PREREQUISITE or gap.confidence >= self.essential_step_threshold:
            gap.importance = StepImportance.ESSENTIAL
        elif gap.gap_type in (GapType.SEMANTIC_DISCONTINUITY, GapType.OBJECT_REFERENCE_GAP):
            gap.importance = StepImportance.RECOMMENDED
        else:
            gap.importance = StepImportance.OPTIONAL

    def _reconstruct_procedure(self, original_steps, gaps):
        sorted_gaps = sorted(gaps, key=lambda g: g.position, reverse=True)
        reconstructed = original_steps.copy()
        for gap in sorted_gaps:
            if gap.inferred_step:
                pos = gap.position + 1 if gap.position >= 0 else 0
                reconstructed.insert(pos, f"[INFERRED] {gap.inferred_step}")
        return reconstructed

    def generate_explanation(self, result: DetectionResult, use_emoji: bool = False) -> str:
        """Generate presentation-ready explanation of detected gaps.

        Args:
            result: DetectionResult to explain
            use_emoji: If True, use emojis (for reports/PPT). If False, use ASCII (for terminals).
        """
        # Emoji vs ASCII markers
        if use_emoji:
            INPUT_ICON = "📥"
            ANALYSIS_ICON = "📊"
            SEARCH_ICON = "🔍"
            ALERT_ICON = "🔔"
            WARNING_ICON = "⚠️"
            CHECK_ICON = "✅"
            OUTPUT_ICON = "📤"
            BRAIN_ICON = "🧠"
        else:
            INPUT_ICON = "[INPUT]"
            ANALYSIS_ICON = "[ANALYSIS]"
            SEARCH_ICON = "[SEARCH]"
            ALERT_ICON = "[!]"
            WARNING_ICON = "[WARNING]"
            CHECK_ICON = "[OK]"
            OUTPUT_ICON = "[OUTPUT]"
            BRAIN_ICON = "[EXPLAIN]"

        lines = [
            "",
            "=" * 70,
            f"{INPUT_ICON} INPUT (Instructional Text)",
            "=" * 70,
            f"Procedure: {result.procedure_title}",
            "",
        ]
        for i, step in enumerate(result.original_steps, 1):
            lines.append(f"Step {i}: {step}")

        # Analysis summary
        lines.extend([
            "",
            "=" * 70,
            f"{ANALYSIS_ICON} ANALYSIS OUTPUT (Intermediate Stage)",
            "=" * 70,
            "[ANALYSIS SUMMARY]",
            f"Total steps detected: {len(result.original_steps)}",
            f"Average step length: {np.mean([len(s.split()) for s in result.original_steps]):.1f} words",
            "",
        ])

        # Step transition analysis
        lines.extend([
            f"{SEARCH_ICON} STEP TRANSITION ANALYSIS",
        ])
        # Show similarity scores between consecutive steps
        for gap in result.detected_gaps:
            if gap.semantic_score > 0:
                pos = gap.position + 1
                lines.append(f"Similarity (Step {pos} -> Step {pos+1}): {gap.semantic_score:.2f}   <-- Low similarity")
        lines.extend([
            "",
            f"{ALERT_ICON} Low similarity indicates a possible missing step.",
            "",
        ])

        # Missing step detection
        if result.detected_gaps:
            lines.extend([
                "=" * 70,
                f"{WARNING_ICON} MISSING / IMPLICIT STEP DETECTION",
                "=" * 70,
            ])
            for gap in result.detected_gaps:
                pos = gap.position + 1
                lines.extend([
                    f"Missing step detected between Step {pos} and Step {pos + 1}",
                    f"Inferred missing step: \"{gap.inferred_step}\"" if gap.inferred_step else "",
                    "Reason:",
                    f"  - {gap.explanation}",
                ])
                for ev in gap.evidence[:2]:
                    lines.append(f"  - {ev}")
                lines.append("")
        else:
            lines.extend([
                "=" * 70,
                f"{CHECK_ICON} NO MISSING STEPS DETECTED",
                "=" * 70,
                "Procedure appears complete and logically consistent.",
                "",
            ])

        # Reconstructed procedure
        lines.extend([
            "=" * 70,
            f"{OUTPUT_ICON} FINAL RECONSTRUCTED INSTRUCTION",
            "=" * 70,
            f"{result.procedure_title} (Completed)",
            "",
        ])
        for i, step in enumerate(result.reconstructed_steps, 1):
            if step.startswith("[INFERRED]"):
                clean_step = step.replace("[INFERRED] ", "")
                lines.append(f"{i}. {clean_step}        [INFERRED]")
            else:
                lines.append(f"{i}. {step}")

        # Explainability
        if result.detected_gaps:
            lines.extend([
                "",
                "=" * 70,
                f"{BRAIN_ICON} EXPLAINABILITY OUTPUT",
                "=" * 70,
            ])
            for gap in result.detected_gaps:
                lines.extend([
                    f"Inferred Step: {gap.inferred_step}" if gap.inferred_step else "",
                    f"Confidence Score: {gap.confidence:.2f}",
                    "Explanation:",
                    f"  - {gap.explanation}",
                ])
                if gap.semantic_score > 0:
                    lines.append(f"  - Low semantic similarity ({gap.semantic_score:.2f}) between adjacent steps")
                lines.append(f"  - Gap type: {gap.gap_type.value.replace('_', ' ').title()}")
                lines.append("")

        lines.append("=" * 70)
        return "\n".join([l for l in lines if l is not None])

    def generate_simple_explanation(self, result: DetectionResult) -> str:
        """Generate simple text explanation (no emojis for terminal compatibility)."""
        lines = [
            f"Missing Step Analysis: {result.procedure_title}",
            "=" * 60,
            f"Original steps: {len(result.original_steps)}",
            f"Gaps detected: {result.num_gaps_detected}",
            f"Essential gaps: {result.num_essential_gaps}",
            f"Avg confidence: {result.avg_confidence:.2%}",
            "",
        ]
        if not result.detected_gaps:
            lines.append("No missing steps detected.")
        else:
            lines.append("Detected Gaps:")
            for i, gap in enumerate(result.detected_gaps, 1):
                lines.extend([
                    f"\n  Gap #{i}: {gap.gap_type.value}",
                    f"    After step {gap.position + 1}",
                    f"    Confidence: {gap.confidence:.2%}",
                    f"    Explanation: {gap.explanation}",
                ])
                if gap.inferred_step:
                    lines.append(f"    Suggested: {gap.inferred_step}")
            lines.extend(["\nReconstructed Procedure:", "-" * 40])
            for i, step in enumerate(result.reconstructed_steps, 1):
                marker = " [NEW]" if step.startswith("[INFERRED]") else ""
                lines.append(f"  {i}. {step.replace('[INFERRED] ', '')}{marker}")
        return "\n".join(lines)


class GapEvaluator:
    """Evaluates gap detection performance."""

    def __init__(self):
        self.predictions, self.ground_truth = [], []

    def add_prediction(self, result: DetectionResult, true_gaps: List[int]):
        self.predictions.append([g.position for g in result.detected_gaps])
        self.ground_truth.append(true_gaps)

    def compute_metrics(self) -> Dict[str, float]:
        tp = fp = fn = 0
        for pred, true in zip(self.predictions, self.ground_truth):
            pred_set, true_set = set(pred), set(true)
            tp += len(pred_set & true_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1,
                "true_positives": tp, "false_positives": fp, "false_negatives": fn}
