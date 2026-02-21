"""
Exploratory and Structural Analysis Module.

Performs MANDATORY analysis before modeling:
- Step length, structure, and ordering pattern analysis
- Prerequisite-action-outcome relationship identification
- Step transition and dependency frequency analysis
- Procedural template detection
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .data_preprocessing import Procedure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StepPattern:
    pattern_type: str
    pattern_text: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class TransitionPattern:
    from_action: str
    to_action: str
    frequency: int
    probability: float
    examples: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class AnalysisResults:
    step_length_stats: Dict[str, float] = field(default_factory=dict)
    steps_per_procedure_stats: Dict[str, float] = field(default_factory=dict)
    action_verb_frequencies: Dict[str, int] = field(default_factory=dict)
    step_patterns: List[StepPattern] = field(default_factory=list)
    transition_patterns: List[TransitionPattern] = field(default_factory=list)
    prerequisite_patterns: List[Dict] = field(default_factory=list)
    outcome_patterns: List[Dict] = field(default_factory=list)
    conditional_patterns: List[Dict] = field(default_factory=list)
    transition_matrix: Optional[np.ndarray] = None
    action_labels: List[str] = field(default_factory=list)
    step_clusters: Dict[int, List[str]] = field(default_factory=dict)
    procedural_templates: List[Dict] = field(default_factory=list)


class ProceduralAnalyzer:
    """Comprehensive analyzer for procedural text."""

    ACTION_VERBS = {
        'add', 'remove', 'insert', 'place', 'put', 'take', 'move', 'turn', 'open', 'close',
        'click', 'select', 'enter', 'press', 'hold', 'mix', 'stir', 'pour', 'heat', 'cook',
        'check', 'verify', 'ensure', 'wait', 'repeat', 'start', 'stop', 'connect', 'install',
        'download', 'save', 'cut', 'fold', 'wrap', 'apply', 'rinse', 'dry', 'clean', 'set',
    }

    PREREQUISITE_INDICATORS = ['before', 'first', 'make sure', 'ensure', 'gather', 'prepare']
    OUTCOME_INDICATORS = ['result', 'should be', 'will be', 'until', 'finally', 'complete']
    CONDITIONAL_INDICATORS = ['if', 'when', 'unless', 'in case', 'should', 'optionally']

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.results = AnalysisResults()
        self.procedures: List[Procedure] = []

    def analyze(self, procedures: List[Procedure]) -> AnalysisResults:
        self.procedures = procedures
        logger.info(f"Analyzing {len(procedures)} procedures")

        self._compute_descriptive_stats()
        self._analyze_action_verbs()
        self._detect_step_patterns()
        self._analyze_transitions()
        self._analyze_relationships()
        self._detect_templates()
        if HAS_SKLEARN:
            self._cluster_steps()

        logger.info("Analysis complete")
        return self.results

    def _compute_descriptive_stats(self):
        step_lengths = [len(s) for p in self.procedures for s in p.steps]
        steps_per_proc = [len(p.steps) for p in self.procedures]

        self.results.step_length_stats = {
            "count": len(step_lengths), "mean": float(np.mean(step_lengths)),
            "std": float(np.std(step_lengths)), "min": int(np.min(step_lengths)),
            "max": int(np.max(step_lengths)), "median": float(np.median(step_lengths)),
        }
        self.results.steps_per_procedure_stats = {
            "count": len(steps_per_proc), "mean": float(np.mean(steps_per_proc)),
            "std": float(np.std(steps_per_proc)), "min": int(np.min(steps_per_proc)),
            "max": int(np.max(steps_per_proc)),
        }

    def _analyze_action_verbs(self):
        verb_counts = Counter()
        for proc in self.procedures:
            for step in proc.steps:
                words = step.lower().split()
                for word in words[:5]:
                    clean = re.sub(r'[^\w]', '', word)
                    if clean in self.ACTION_VERBS:
                        verb_counts[clean] += 1
                        break
        self.results.action_verb_frequencies = dict(verb_counts.most_common(50))

    def _detect_step_patterns(self):
        patterns = []
        total_steps = sum(len(p.steps) for p in self.procedures)

        # Check pattern types
        for pattern_type, indicators in [
            ("prerequisite", self.PREREQUISITE_INDICATORS),
            ("outcome", self.OUTCOME_INDICATORS),
            ("conditional", self.CONDITIONAL_INDICATORS)
        ]:
            matching = []
            for proc in self.procedures:
                for step in proc.steps:
                    if any(ind in step.lower() for ind in indicators):
                        matching.append(step)
            if matching:
                patterns.append(StepPattern(
                    pattern_type=pattern_type,
                    pattern_text=f"Steps with {pattern_type} indicators",
                    frequency=len(matching),
                    examples=matching[:5],
                    confidence=len(matching) / total_steps
                ))

        # Imperative pattern
        imperative_count = sum(
            1 for p in self.procedures for s in p.steps
            if s.split()[0].lower().rstrip('.,') in self.ACTION_VERBS
        )
        patterns.append(StepPattern(
            pattern_type="imperative", pattern_text="Steps starting with action verbs",
            frequency=imperative_count, confidence=imperative_count / total_steps
        ))
        self.results.step_patterns = patterns

    def _analyze_transitions(self):
        def extract_action(step):
            for word in step.lower().split()[:5]:
                clean = re.sub(r'[^\w]', '', word)
                if clean in self.ACTION_VERBS:
                    return clean
            return "other"

        transition_counts = defaultdict(int)
        action_counts = Counter()

        for proc in self.procedures:
            for i in range(len(proc.steps) - 1):
                a1, a2 = extract_action(proc.steps[i]), extract_action(proc.steps[i + 1])
                action_counts[a1] += 1
                transition_counts[(a1, a2)] += 1

        transition_patterns = []
        for (a1, a2), count in sorted(transition_counts.items(), key=lambda x: -x[1])[:100]:
            prob = count / action_counts[a1] if action_counts[a1] > 0 else 0
            transition_patterns.append(TransitionPattern(a1, a2, count, prob))
        self.results.transition_patterns = transition_patterns

        # Build matrix
        actions = sorted(set(action_counts.keys()))
        n = len(actions)
        action_idx = {a: i for i, a in enumerate(actions)}
        matrix = np.zeros((n, n))
        for (a1, a2), count in transition_counts.items():
            if a1 in action_idx and a2 in action_idx:
                matrix[action_idx[a1], action_idx[a2]] = count
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.results.transition_matrix = matrix / row_sums
        self.results.action_labels = actions

    def _analyze_relationships(self):
        for proc in self.procedures:
            if proc.steps:
                first_lower = proc.steps[0].lower()
                if any(ind in first_lower for ind in self.PREREQUISITE_INDICATORS):
                    self.results.prerequisite_patterns.append({
                        "procedure": proc.title, "step": proc.steps[0], "position": "first"
                    })
                last_lower = proc.steps[-1].lower()
                if any(ind in last_lower for ind in self.OUTCOME_INDICATORS):
                    self.results.outcome_patterns.append({
                        "procedure": proc.title, "step": proc.steps[-1], "position": "last"
                    })

    def _detect_templates(self):
        def classify_step(step):
            s = step.lower()
            if any(ind in s for ind in self.PREREQUISITE_INDICATORS): return "P"
            if any(ind in s for ind in self.OUTCOME_INDICATORS): return "O"
            if any(ind in s for ind in self.CONDITIONAL_INDICATORS): return "C"
            return "A"

        signatures = Counter()
        for proc in self.procedures:
            sig = "".join(classify_step(s) for s in proc.steps)
            signatures[sig] += 1

        self.results.procedural_templates = [
            {"signature": sig, "frequency": cnt, "length": len(sig)}
            for sig, cnt in signatures.most_common(20) if cnt >= 2
        ]

    def _cluster_steps(self):
        all_steps = [s for p in self.procedures for s in p.steps]
        if len(all_steps) < 10:
            return
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf = vectorizer.fit_transform(all_steps)
            n_clusters = min(10, len(all_steps) // 5)
            kmeans = KMeans(n_clusters=max(2, n_clusters), random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf)
            clusters = defaultdict(list)
            for step, label in zip(all_steps, labels):
                if len(clusters[label]) < 10:
                    clusters[label].append(step)
            self.results.step_clusters = dict(clusters)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")

    def generate_report(self, output_path: Optional[str] = None) -> str:
        lines = [
            "=" * 60, "PROCEDURAL TEXT ANALYSIS REPORT", "=" * 60, "",
            "1. DESCRIPTIVE STATISTICS", "-" * 40,
            f"Step count: {self.results.step_length_stats.get('count', 0)}",
            f"Mean step length: {self.results.step_length_stats.get('mean', 0):.1f}",
            f"Mean steps per procedure: {self.results.steps_per_procedure_stats.get('mean', 0):.1f}",
            "", "2. TOP ACTION VERBS", "-" * 40,
        ]
        for verb, cnt in list(self.results.action_verb_frequencies.items())[:10]:
            lines.append(f"  {verb}: {cnt}")
        lines.extend(["", "3. STEP PATTERNS", "-" * 40])
        for p in self.results.step_patterns:
            lines.append(f"  {p.pattern_type}: {p.frequency} ({p.confidence:.1%})")
        lines.extend(["", "4. TOP TRANSITIONS", "-" * 40])
        for t in self.results.transition_patterns[:10]:
            lines.append(f"  {t.from_action} -> {t.to_action}: {t.frequency} ({t.probability:.1%})")
        lines.extend(["", "=" * 60])

        report = "\n".join(lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

    def generate_transition_table(self) -> str:
        if self.results.transition_matrix is None:
            return "No transition matrix available"
        labels = self.results.action_labels[:8]
        header = "From\\To".ljust(10) + " ".join(l[:7].ljust(7) for l in labels)
        rows = [header, "-" * len(header)]
        for i, from_a in enumerate(labels):
            vals = " ".join(f"{self.results.transition_matrix[i,j]:.2f}".ljust(7) for j in range(len(labels)))
            rows.append(f"{from_a[:8].ljust(10)}{vals}")
        return "\n".join(rows)

    def save_results(self, output_path: str):
        data = {
            "step_length_stats": self.results.step_length_stats,
            "steps_per_procedure_stats": self.results.steps_per_procedure_stats,
            "action_verb_frequencies": self.results.action_verb_frequencies,
            "transition_patterns": [{"from": t.from_action, "to": t.to_action, "freq": t.frequency, "prob": t.probability}
                                    for t in self.results.transition_patterns[:50]],
            "procedural_templates": self.results.procedural_templates,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def plot_analysis(self, output_dir: str):
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Step length histogram
            step_lengths = [len(s) for p in self.procedures for s in p.steps]
            plt.figure(figsize=(10, 6))
            plt.hist(step_lengths, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Step Length')
            plt.ylabel('Frequency')
            plt.title('Step Length Distribution')
            plt.savefig(f"{output_dir}/step_lengths.png", dpi=150)
            plt.close()

            # Action verb bar chart
            verbs = list(self.results.action_verb_frequencies.keys())[:15]
            counts = [self.results.action_verb_frequencies[v] for v in verbs]
            plt.figure(figsize=(12, 6))
            plt.barh(verbs, counts, color='steelblue')
            plt.xlabel('Frequency')
            plt.title('Top Action Verbs')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/action_verbs.png", dpi=150)
            plt.close()

            logger.info(f"Plots saved to {output_dir}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
