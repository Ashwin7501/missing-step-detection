"""
Data Preprocessing Module for Missing Step Detection System.

Dataset Used: WikiHow (or synthetic for demonstration)
- WikiHow: Large-scale dataset of how-to articles with step-by-step instructions
- Synthetic: Built-in procedural instructions for testing
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Procedure:
    """Represents a single procedure with its steps."""
    id: str
    title: str
    category: str = ""
    steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "steps": self.steps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Procedure":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            category=data.get("category", ""),
            steps=data.get("steps", []),
            metadata=data.get("metadata", {}),
        )


class DataPreprocessor:
    """Handles loading, cleaning, and preprocessing of procedural text data."""

    def __init__(self, config: Dict):
        self.config = config
        self.procedures: List[Procedure] = []
        self.stats: Dict[str, Any] = {}
        self._compile_patterns()

    def _compile_patterns(self):
        self.url_pattern = re.compile(r'http[s]?://[^\s]+')
        self.step_number_pattern = re.compile(r'^\s*(?:\d+[\.\):]|\*|\-|\•)\s*')
        self.whitespace_pattern = re.compile(r'\s+')

    def load_dataset(self, source: str = "synthetic", dataset_name: str = "wikihow",
                     subset: str = "all", max_samples: Optional[int] = None,
                     local_path: Optional[str] = None) -> List[Procedure]:
        logger.info(f"Loading dataset from {source}")

        if source == "huggingface":
            self.procedures = self._load_from_huggingface(dataset_name, subset, max_samples)
        elif source == "local" and local_path:
            self.procedures = self._load_from_local(local_path, max_samples)
        else:
            self.procedures = self._create_synthetic_dataset(max_samples or 100)

        self.procedures = self._preprocess_procedures(self.procedures)
        self.stats = self._compute_statistics()
        logger.info(f"Loaded {len(self.procedures)} procedures")
        return self.procedures

    def _load_from_huggingface(self, dataset_name, subset, max_samples):
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, subset, split="train", trust_remote_code=True)
            procedures = []
            samples = list(dataset)[:max_samples] if max_samples else list(dataset)
            for idx, item in enumerate(samples):
                text = item.get("text", "") or item.get("article", "")
                title = item.get("title", f"Procedure_{idx}")
                steps = self._extract_steps_from_text(text)
                if len(steps) >= 3:
                    procedures.append(Procedure(id=f"wikihow_{idx}", title=title, steps=steps))
            return procedures
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}")
            return self._create_synthetic_dataset(max_samples or 100)

    def _extract_steps_from_text(self, text: str) -> List[str]:
        steps = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                clean_line = self.step_number_pattern.sub('', line).strip()
                if clean_line:
                    steps.append(clean_line)
        return [s for s in steps if 5 <= len(s) <= 500]

    def _load_from_local(self, local_path, max_samples):
        path = Path(local_path)
        procedures = []
        if path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("procedures", [])
            for item in items[:max_samples]:
                procedures.append(Procedure.from_dict(item))
        return procedures

    def _create_synthetic_dataset(self, n_samples: int) -> List[Procedure]:
        logger.info(f"Creating synthetic dataset with {n_samples} samples")
        templates = [
            {"title": "How to Make Scrambled Eggs", "category": "cooking",
             "steps": ["Crack 2-3 eggs into a bowl.", "Add salt and pepper to taste.",
                       "Whisk until fully combined.", "Heat a non-stick pan over medium heat.",
                       "Add butter and let it melt.", "Pour the egg mixture into the pan.",
                       "Stir gently as eggs begin to set.", "Remove from heat while slightly wet.",
                       "Serve immediately on a warm plate."]},
            {"title": "How to Set Up a Smartphone", "category": "technology",
             "steps": ["Remove phone from packaging.", "Insert the SIM card.",
                       "Press and hold power button to turn on.", "Select your language.",
                       "Connect to a WiFi network.", "Sign in with your account.",
                       "Set up screen lock security.", "Install essential apps."]},
            {"title": "How to Change a Light Bulb", "category": "home",
             "steps": ["Turn off the light switch.", "Wait for bulb to cool.",
                       "Position a ladder beneath the fixture.", "Grip old bulb and turn counterclockwise.",
                       "Remove the old bulb.", "Insert new bulb and turn clockwise.",
                       "Turn on the switch to test."]},
            {"title": "How to Brush Teeth Properly", "category": "health",
             "steps": ["Wet your toothbrush.", "Apply pea-sized amount of toothpaste.",
                       "Place brush at 45-degree angle to gums.", "Use short back-and-forth strokes.",
                       "Brush outer surfaces of all teeth.", "Brush inner surfaces.",
                       "Brush chewing surfaces.", "Brush your tongue.",
                       "Spit out toothpaste and rinse."]},
            {"title": "How to Send an Email", "category": "office",
             "steps": ["Open your email application.", "Click Compose or New Message.",
                       "Enter recipient's email address.", "Add a clear subject line.",
                       "Write your message in the body.", "Review for spelling errors.",
                       "Add attachments if needed.", "Click Send."]},
        ]
        procedures = []
        for i in range(n_samples):
            t = templates[i % len(templates)]
            procedures.append(Procedure(
                id=f"synthetic_{i}", title=t["title"],
                category=t["category"], steps=t["steps"].copy()
            ))
        return procedures

    def _preprocess_procedures(self, procedures):
        processed = []
        for proc in procedures:
            valid_steps = [self._clean_step(s) for s in proc.steps]
            valid_steps = [s for s in valid_steps if 5 <= len(s) <= 500]
            if 3 <= len(valid_steps) <= 30:
                proc.steps = valid_steps
                processed.append(proc)
        return processed

    def _clean_step(self, text):
        text = self.url_pattern.sub('', text)
        text = self.step_number_pattern.sub('', text)
        return self.whitespace_pattern.sub(' ', text).strip()

    def _compute_statistics(self):
        if not self.procedures:
            return {}
        step_counts = [len(p.steps) for p in self.procedures]
        step_lengths = [len(s) for p in self.procedures for s in p.steps]
        return {
            "num_procedures": len(self.procedures),
            "total_steps": sum(step_counts),
            "avg_steps_per_procedure": np.mean(step_counts),
            "avg_step_length": np.mean(step_lengths),
        }

    def create_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        if not HAS_SKLEARN:
            n = len(self.procedures)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            return self.procedures[:train_end], self.procedures[train_end:val_end], self.procedures[val_end:]
        train, temp = train_test_split(self.procedures, train_size=train_ratio, random_state=random_seed)
        val_adj = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(temp, train_size=val_adj, random_state=random_seed)
        return train, val, test

    def save_dataset(self, output_path: str, procedures: List[Procedure] = None):
        procs = procedures or self.procedures
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"procedures": [p.to_dict() for p in procs], "statistics": self.stats}, f, indent=2)


class ProcedureDataset:
    """PyTorch-compatible Dataset for procedural text."""
    def __init__(self, procedures: List[Procedure], tokenizer=None, max_length=256, mode="sequence"):
        self.procedures = procedures
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        if self.mode == "sequence":
            for proc in self.procedures:
                samples.append({"procedure_id": proc.id, "steps": proc.steps})
        elif self.mode == "pairs":
            for proc in self.procedures:
                for i in range(len(proc.steps) - 1):
                    samples.append({"step1": proc.steps[i], "step2": proc.steps[i+1], "is_consecutive": True})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_negative_samples(procedures, n_negatives_per_positive=1, random_seed=42):
    np.random.seed(random_seed)
    all_steps = [(p.id, s) for p in procedures for s in p.steps]
    negatives = []
    for proc in procedures:
        for i in range(len(proc.steps) - 1):
            for _ in range(n_negatives_per_positive):
                _, rand_step = all_steps[np.random.randint(len(all_steps))]
                if rand_step != proc.steps[i + 1]:
                    negatives.append({"current_step": proc.steps[i], "next_step": rand_step, "is_consecutive": False})
    return negatives
