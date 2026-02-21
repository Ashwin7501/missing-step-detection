# Missing Step Detection System

An end-to-end NLP system for automatic detection and inference of missing and implicit steps in instructional/procedural text.

## Overview

This system analyzes procedural documents (user manuals, SOPs, online guides, technical documentation) to:
- **Identify gaps** in step sequences
- **Detect unstated prerequisites** and implicitly assumed actions
- **Reconstruct complete, logically consistent procedures**
- **Provide explainable reasoning** for why steps are inferred as missing

## Features

- **Multi-method detection**: Combines semantic similarity, transition probability, and action-object coherence analysis
- **Transformer-based models**: Uses BERT for step encoding and semantic understanding
- **Sequence modeling**: BiLSTM/Transformer for capturing procedural flow
- **Comprehensive analysis**: Step patterns, transition frequencies, procedural templates
- **Explainable AI**: Clear explanations for each detected gap
- **Modular architecture**: Easy to extend and customize

## Project Structure

```
missing_step_detection/
├── config.py                 # Configuration settings
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Dataset loading and parsing
│   ├── analysis.py           # Exploratory and structural analysis
│   ├── step_extraction.py    # Action/entity extraction
│   ├── sequence_model.py     # Transformer and sequence models
│   ├── missing_step_detection.py  # Gap detection logic
│   └── inference.py          # End-to-end inference pipeline
├── data/                     # Dataset storage
├── models/                   # Trained model checkpoints
└── outputs/                  # Results and reports
```

## Dataset

The system supports multiple data sources:

### 1. WikiHow Dataset (Default)
- Large-scale dataset of how-to articles with step-by-step instructions
- ~230k articles across various domains (cooking, technology, health, etc.)
- Loaded from HuggingFace datasets library

### 2. Synthetic Dataset (Built-in)
- Realistic procedural instructions for testing
- Covers multiple domains: cooking, technology, home repair, personal care, office
- Used when WikiHow is unavailable or for quick demonstrations

### 3. Custom Data
- JSON format with procedure objects
- Plain text files with steps (one per line)

## Installation

### 1. Clone/Download the Project
```bash
cd missing_step_detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. (Optional) Install PyTorch with CUDA
For GPU acceleration:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## How to Run

### Quick Demo
```bash
python main.py --mode demo
```

This runs a complete demonstration including:
- Loading synthetic dataset
- Running exploratory analysis
- Detecting missing steps
- Generating reports

### Run Exploratory Analysis Only
```bash
python main.py --mode analyze
```

Outputs:
- Descriptive statistics
- Transition frequency tables
- Step pattern analysis
- Visualization plots

### Run Detection Pipeline
```bash
# On synthetic data
python main.py --mode detect --max-samples 100

# On custom input file
python main.py --mode detect --input your_procedure.json
```

### Train Models
```bash
python main.py --mode train
```

Trains:
- Transition predictor (BERT + classifier)
- Sequence model (BiLSTM)

### Evaluate Performance
```bash
python main.py --mode evaluate
```

Computes:
- Precision, Recall, F1-score
- Qualitative examples
- Error analysis

## Input Formats

### JSON Format
```json
{
  "id": "procedure_001",
  "title": "How to Make Tea",
  "category": "cooking",
  "steps": [
    "Fill the kettle with water.",
    "Boil the water.",
    "Add tea bag to cup.",
    "Pour hot water over tea bag.",
    "Steep for 3-5 minutes.",
    "Remove tea bag and enjoy."
  ]
}
```

### Text Format (steps.txt)
```
Fill the kettle with water.
Boil the water.
Add tea bag to cup.
Pour hot water over tea bag.
Steep for 3-5 minutes.
Remove tea bag and enjoy.
```

## Output Format

### Detection Results (JSON)
```json
{
  "procedure_id": "test_001",
  "procedure_title": "How to Make Tea",
  "original_steps": ["..."],
  "detected_gaps": [
    {
      "position": 2,
      "before_step": "Boil the water.",
      "after_step": "Pour hot water over tea bag.",
      "gap_type": "missing_intermediate",
      "importance": "essential",
      "confidence": 0.85,
      "inferred_step": "Add tea bag to cup.",
      "explanation": "Missing step to prepare cup before pouring water",
      "evidence": [
        "Large semantic gap detected",
        "Object 'tea bag' referenced without introduction"
      ]
    }
  ],
  "reconstructed_steps": ["..."],
  "num_gaps_detected": 1,
  "avg_confidence": 0.85
}
```

### Analysis Report (Text)
```
======================================================================
PROCEDURAL TEXT ANALYSIS REPORT
======================================================================

1. DESCRIPTIVE STATISTICS
----------------------------------------
Step Length Statistics:
  count: 500
  mean: 45.3
  std: 18.2
  ...

2. ACTION VERB ANALYSIS
----------------------------------------
Top 20 Action Verbs:
  add: 120
  remove: 85
  click: 72
  ...

3. TRANSITION ANALYSIS
----------------------------------------
Top 15 Step Transitions:
  add -> mix: count=45, prob=0.23
  heat -> add: count=38, prob=0.19
  ...
```

## Configuration

Edit `config.py` to customize:

```python
# Detection thresholds
DETECTION_CONFIG = {
    "semantic_gap_threshold": 0.4,      # Below this = potential gap
    "transition_anomaly_threshold": 0.3, # Below this = unusual transition
    "min_confidence_threshold": 0.5,     # Minimum to report
    "essential_step_threshold": 0.7,     # For importance classification
}

# Model settings
MODEL_CONFIG = {
    "transformer_model": "bert-base-uncased",
    "max_seq_length": 256,
    "lstm_hidden_size": 256,
    "learning_rate": 2e-5,
    "num_epochs": 10,
}
```

## Detection Methods

### 1. Semantic Similarity Analysis
- Computes semantic similarity between consecutive steps
- Flags transitions with unusually low similarity
- Uses sentence-transformers for embeddings

### 2. Transition Probability Analysis
- Learns common action transitions from training data
- Flags transitions with low probability
- Suggests intermediate actions

### 3. Action-Object Coherence
- Tracks object introductions and references
- Detects objects used without introduction
- Identifies prerequisite gaps

### 4. Model-Based Detection (Optional)
- BERT-encoded step representations
- Trained classifier for transition validity
- Confidence estimation

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Precision** | Correct gaps / All detected gaps |
| **Recall** | Correct gaps / All true gaps |
| **F1-Score** | Harmonic mean of precision and recall |

## Example Output

**Input Procedure: "How to Change a Flat Tire"**
```
1. Pull over to a safe location.
2. Get the spare tire and jack from the trunk.
3. Position the jack under the vehicle.
4. Raise the vehicle.
5. Remove the lug nuts.
6. Mount the spare tire.
7. Lower the vehicle.
```

**Detected Gaps:**
```
Gap #1: After step 1
  Type: missing_prerequisite
  Confidence: 82%
  Suggested: "Turn on hazard lights and apply parking brake."

Gap #2: After step 2
  Type: missing_intermediate
  Confidence: 75%
  Suggested: "Loosen lug nuts before raising vehicle."

Gap #3: After step 5
  Type: action_sequence_gap
  Confidence: 70%
  Suggested: "Remove the flat tire from the wheel."
```

## API Usage

```python
from src.inference import InferencePipeline, PipelineConfig
from src.data_preprocessing import Procedure

# Initialize pipeline
config = PipelineConfig(
    run_analysis=True,
    verbose=True,
)
pipeline = InferencePipeline(config)

# Process single procedure
procedure = Procedure(
    id="my_proc",
    title="How to Make Coffee",
    steps=[
        "Fill kettle with water.",
        "Pour water into cup with coffee.",  # Missing: boil water
        "Stir and enjoy.",
    ]
)

result = pipeline.run_single(procedure)

# Get explanation
explanation = pipeline.explain_detection(result)
print(explanation)

# Access detected gaps
for gap in result.detected_gaps:
    print(f"Position {gap.position}: {gap.inferred_step}")
    print(f"  Confidence: {gap.confidence:.2%}")
    print(f"  Reason: {gap.explanation}")
```

## Extending the System

### Adding New Detection Methods
```python
# In missing_step_detection.py

def _detect_custom_gaps(self, structured_steps):
    gaps = []
    # Your detection logic here
    return gaps

# Add to detect_missing_steps()
if 'custom' in methods:
    custom_gaps = self._detect_custom_gaps(structured_steps)
    all_gaps.extend(custom_gaps)
```

### Adding New Step Patterns
```python
# In analysis.py

CUSTOM_INDICATORS = ['custom_keyword', 'another_keyword']

# Add pattern detection in _detect_step_patterns()
```

## Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **CUDA out of memory**
   - Reduce batch size in `config.py`
   - Use CPU: set `device='cpu'` in trainer

3. **WikiHow dataset unavailable**
   - System automatically falls back to synthetic data
   - Or specify `source="synthetic"` explicitly

4. **transformers import error**
   ```bash
   pip install transformers>=4.20.0
   ```

## Citation

If you use this system in your research, please cite:
```bibtex
@software{missing_step_detection,
  title={Missing Step Detection System for Procedural Text},
  year={2024},
  description={NLP system for detecting implicit steps in instructions}
}
```

## License

This project is for educational and research purposes.

## Acknowledgments

- WikiHow for procedural text data
- HuggingFace for transformers and datasets
- spaCy for NLP tools
- sentence-transformers for semantic embeddings
