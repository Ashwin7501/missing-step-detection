"""
Step Extraction Module - Extract actions, objects, conditions from steps.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

from .data_preprocessing import Procedure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StepComponent:
    text: str
    component_type: str
    confidence: float = 1.0


@dataclass
class StructuredStep:
    original_text: str
    step_index: int = 0
    main_action: Optional[StepComponent] = None
    primary_object: Optional[StepComponent] = None
    preconditions: List[StepComponent] = field(default_factory=list)
    temporal_markers: List[StepComponent] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    is_imperative: bool = False
    is_conditional: bool = False
    has_prerequisite: bool = False

    def get_action_object_pair(self) -> Tuple[str, str]:
        action = self.main_action.text if self.main_action else ""
        obj = self.primary_object.text if self.primary_object else ""
        return (action, obj)


class StepExtractor:
    """Extracts structured information from procedural text steps."""

    ACTION_VERBS = {
        # Basic manipulation
        'add', 'remove', 'insert', 'place', 'put', 'take', 'move', 'turn', 'open', 'close',
        'get', 'grab', 'lift', 'pull', 'push', 'raise', 'lower',
        # Computer/Tech
        'click', 'select', 'enter', 'press', 'hold', 'type', 'tap', 'scroll',
        'download', 'save', 'install', 'run', 'launch', 'execute', 'upload', 'delete',
        'connect', 'disconnect', 'start', 'stop', 'restart', 'update',
        # Cooking
        'mix', 'stir', 'pour', 'heat', 'cook', 'bake', 'fry', 'boil', 'simmer',
        'crack', 'whisk', 'season', 'serve', 'peel', 'chop', 'slice', 'dice', 'mince',
        'blend', 'fold', 'wrap', 'spread', 'drain', 'strain', 'fill', 'empty',
        'preheat', 'grease', 'marinate', 'grill', 'roast', 'toast',
        # Cleaning/Hygiene
        'wash', 'rinse', 'dry', 'clean', 'wipe', 'scrub', 'rub', 'wet', 'soak',
        'apply', 'spray', 'polish', 'sweep', 'mop', 'vacuum',
        # General
        'check', 'verify', 'ensure', 'wait', 'repeat', 'set', 'adjust', 'measure',
        'cut', 'attach', 'detach', 'secure', 'loosen', 'tighten',
        'read', 'write', 'note', 'record', 'mark',
        'drive', 'walk', 'go', 'come', 'bring', 'carry',
        'use', 'make', 'create', 'build', 'prepare', 'finish', 'complete',
    }

    TEMPORAL_MARKERS = {'before', 'after', 'while', 'during', 'until', 'when', 'first', 'then', 'finally'}
    CONDITION_INDICATORS = {'if', 'unless', 'when', 'provided', 'should'}
    PREREQUISITE_INDICATORS = {'make sure', 'ensure', 'verify', 'before', 'first', 'must have'}

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sentence_model = None
        self._load_sentence_model()

    def _load_sentence_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get("sentence_transformer", "all-MiniLM-L6-v2")
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed")
        except Exception as e:
            logger.warning(f"Could not load sentence model: {e}")

    def extract_step(self, step_text: str, step_index: int = 0) -> StructuredStep:
        structured = StructuredStep(original_text=step_text, step_index=step_index)

        # Extract action verb
        words = step_text.split()
        for i, word in enumerate(words[:5]):
            clean = re.sub(r'[^\w]', '', word.lower())
            if clean in self.ACTION_VERBS:
                structured.main_action = StepComponent(text=clean, component_type="action")
                # Extract object (words after verb)
                if i + 1 < len(words):
                    obj_words = []
                    for w in words[i+1:i+5]:
                        w_clean = w.strip('.,;!?')
                        if w_clean.lower() not in ('the', 'a', 'an', 'your'):
                            obj_words.append(w_clean)
                        if w_clean.endswith('.'):
                            break
                    if obj_words:
                        structured.primary_object = StepComponent(
                            text=" ".join(obj_words), component_type="object"
                        )
                break

        # Check flags
        step_lower = step_text.lower()
        structured.is_imperative = any(step_text.split()[0].lower().rstrip('.,') == v for v in self.ACTION_VERBS) if words else False
        structured.is_conditional = any(ind in step_lower for ind in self.CONDITION_INDICATORS)
        structured.has_prerequisite = any(ind in step_lower for ind in self.PREREQUISITE_INDICATORS)

        # Extract temporal markers
        for word in words:
            if word.lower().strip('.,') in self.TEMPORAL_MARKERS:
                structured.temporal_markers.append(StepComponent(text=word.lower(), component_type="temporal"))

        # Generate embedding
        if self.sentence_model:
            structured.embedding = self.sentence_model.encode(step_text)

        return structured

    def extract_procedure(self, procedure: Procedure) -> List[StructuredStep]:
        return [self.extract_step(step, i) for i, step in enumerate(procedure.steps)]

    def compute_similarity(self, step1: StructuredStep, step2: StructuredStep) -> float:
        if step1.embedding is not None and step2.embedding is not None:
            # Check if embeddings are valid (not zeros)
            norm1, norm2 = np.linalg.norm(step1.embedding), np.linalg.norm(step2.embedding)
            if norm1 > 0.01 and norm2 > 0.01:
                dot = np.dot(step1.embedding, step2.embedding)
                return float(dot / (norm1 * norm2))

        # Improved fallback: word overlap + action-object + semantic relatedness
        return self._compute_fallback_similarity(step1, step2)

    def _compute_fallback_similarity(self, step1: StructuredStep, step2: StructuredStep) -> float:
        """Compute similarity when embeddings are not available."""
        score = 0.0

        # 1. Action matching (0.3 weight)
        a1, o1 = step1.get_action_object_pair()
        a2, o2 = step2.get_action_object_pair()
        if a1 and a2:
            if a1 == a2:
                score += 0.3
            elif self._are_related_actions(a1, a2):
                score += 0.2

        # 2. Object overlap (0.3 weight)
        if o1 and o2:
            words1 = set(w.lower() for w in o1.split() if len(w) > 2)
            words2 = set(w.lower() for w in o2.split() if len(w) > 2)
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                score += 0.3 * overlap

        # 3. Full text word overlap (0.4 weight)
        text1 = step1.original_text.lower()
        text2 = step2.original_text.lower()

        # Remove common stopwords and short words
        stopwords = {'the', 'a', 'an', 'to', 'and', 'or', 'in', 'on', 'at', 'for', 'with', 'your', 'you', 'it', 'is', 'be'}
        words1 = set(w.strip('.,;!?()') for w in text1.split() if w not in stopwords and len(w) > 2)
        words2 = set(w.strip('.,;!?()') for w in text2.split() if w not in stopwords and len(w) > 2)

        if words1 and words2:
            # Jaccard similarity
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            score += 0.4 * overlap

        return min(score, 1.0)

    def _are_related_actions(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically related."""
        related_groups = [
            {'add', 'put', 'place', 'insert', 'pour'},
            {'remove', 'take', 'get', 'pull'},
            {'mix', 'stir', 'whisk', 'blend'},
            {'heat', 'cook', 'boil', 'warm'},
            {'cut', 'slice', 'chop', 'dice'},
            {'wash', 'rinse', 'clean'},
            {'dry', 'wipe', 'towel'},
            {'open', 'start', 'begin', 'turn'},
            {'close', 'stop', 'end', 'finish'},
            {'click', 'select', 'choose', 'press'},
            {'enter', 'type', 'input', 'write'},
            {'download', 'install', 'get'},
            {'run', 'execute', 'start', 'launch'},
            {'wait', 'pause', 'hold'},
            {'check', 'verify', 'ensure', 'confirm'},
        ]
        for group in related_groups:
            if action1 in group and action2 in group:
                return True
        return False

    def compute_step_embeddings(self, steps: List[str], batch_size: int = 32) -> np.ndarray:
        if self.sentence_model is None:
            return np.zeros((len(steps), 384))
        return self.sentence_model.encode(steps, batch_size=batch_size)
