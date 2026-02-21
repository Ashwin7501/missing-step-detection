"""
Inference Pipeline Module - End-to-end pipeline for missing step detection.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .data_preprocessing import DataPreprocessor, Procedure
from .analysis import ProceduralAnalyzer, AnalysisResults
from .step_extraction import StepExtractor
from .missing_step_detection import MissingStepDetector, DetectionResult, GapEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    use_synthetic_data: bool = True
    max_samples: int = 100
    run_analysis: bool = True
    save_analysis: bool = True
    detection_methods: List[str] = field(default_factory=lambda: ['semantic', 'transition', 'coherence'])
    use_model: bool = False
    model_path: Optional[str] = None
    output_dir: str = "outputs"
    save_results: bool = True
    generate_report: bool = True
    verbose: bool = True


@dataclass
class PipelineOutput:
    detection_results: List[DetectionResult]
    analysis_results: Optional[AnalysisResults] = None
    analysis_report: str = ""
    evaluation_metrics: Optional[Dict[str, float]] = None
    total_procedures: int = 0
    total_gaps_detected: int = 0
    total_essential_gaps: int = 0
    avg_gaps_per_procedure: float = 0.0
    avg_confidence: float = 0.0
    total_time: float = 0.0
    analysis_time: float = 0.0
    detection_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "detection_results": [r.to_dict() for r in self.detection_results],
            "evaluation_metrics": self.evaluation_metrics,
            "summary": {
                "total_procedures": self.total_procedures,
                "total_gaps_detected": self.total_gaps_detected,
                "avg_gaps_per_procedure": self.avg_gaps_per_procedure,
            },
            "timing": {"total_time": self.total_time, "analysis_time": self.analysis_time},
        }


class InferencePipeline:
    """End-to-end inference pipeline for missing step detection."""

    def __init__(self, config: Union[PipelineConfig, Dict] = None,
                 preprocessing_config: Dict = None, analysis_config: Dict = None,
                 nlp_config: Dict = None, detection_config: Dict = None, model_config: Dict = None):
        if isinstance(config, dict):
            config = PipelineConfig(**config)
        self.config = config or PipelineConfig()
        self.preprocessing_config = preprocessing_config or {}
        self.analysis_config = analysis_config or {}
        self.nlp_config = nlp_config or {}
        self.detection_config = detection_config or {}

        self.preprocessor = DataPreprocessor(self.preprocessing_config)
        self.analyzer = ProceduralAnalyzer(self.analysis_config)
        self.step_extractor = StepExtractor(self.nlp_config)
        self.detector = None
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, procedures: List[Procedure] = None, data_source: str = "synthetic",
            ground_truth: Dict[str, List[int]] = None) -> PipelineOutput:
        start_time = time.time()

        # Load data
        if procedures is None:
            logger.info("Loading dataset...")
            procedures = self.preprocessor.load_dataset(source=data_source, max_samples=self.config.max_samples)
            logger.info(f"Loaded {len(procedures)} procedures")

        # Analysis
        analysis_results, analysis_report, analysis_time = None, "", 0.0
        if self.config.run_analysis:
            logger.info("Running analysis...")
            analysis_start = time.time()
            analysis_results = self.analyzer.analyze(procedures)
            analysis_report = self.analyzer.generate_report()
            analysis_time = time.time() - analysis_start
            if self.config.save_analysis:
                self._save_analysis(analysis_results, analysis_report)
            if self.config.verbose:
                print("\n" + "=" * 60)
                print("ANALYSIS SUMMARY")
                print(f"Total procedures: {len(procedures)}")
                print(f"Avg steps per procedure: {analysis_results.steps_per_procedure_stats.get('mean', 0):.1f}")

        # Initialize detector
        self.detector = MissingStepDetector(
            config=self.detection_config,
            step_extractor=self.step_extractor,
            analyzer=self.analyzer if analysis_results else None,
        )

        # Detection
        logger.info("Running detection...")
        detection_start = time.time()
        detection_results = []
        evaluator = GapEvaluator() if ground_truth else None

        for i, proc in enumerate(procedures):
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(procedures)}...")
            result = self.detector.detect_missing_steps(proc, methods=self.config.detection_methods)
            detection_results.append(result)
            if evaluator and proc.id in ground_truth:
                evaluator.add_prediction(result, ground_truth[proc.id])

        detection_time = time.time() - detection_start
        evaluation_metrics = evaluator.compute_metrics() if evaluator else None

        # Summary
        total_gaps = sum(r.num_gaps_detected for r in detection_results)
        total_essential = sum(r.num_essential_gaps for r in detection_results)
        avg_gaps = total_gaps / len(detection_results) if detection_results else 0
        confidences = [r.avg_confidence for r in detection_results if r.avg_confidence > 0]
        avg_conf = np.mean(confidences) if confidences else 0

        output = PipelineOutput(
            detection_results=detection_results, analysis_results=analysis_results,
            analysis_report=analysis_report, evaluation_metrics=evaluation_metrics,
            total_procedures=len(procedures), total_gaps_detected=total_gaps,
            total_essential_gaps=total_essential, avg_gaps_per_procedure=avg_gaps,
            avg_confidence=avg_conf, total_time=time.time() - start_time,
            analysis_time=analysis_time, detection_time=detection_time,
        )

        if self.config.save_results:
            self._save_results(output)
        if self.config.generate_report:
            report = self._generate_report(output)
            if self.config.verbose:
                print(report)

        logger.info(f"Pipeline completed in {output.total_time:.2f}s")
        return output

    def run_single(self, procedure: Procedure) -> DetectionResult:
        if self.detector is None:
            self.detector = MissingStepDetector(config=self.detection_config, step_extractor=self.step_extractor)
        return self.detector.detect_missing_steps(procedure, methods=self.config.detection_methods)

    def run_from_text(self, title: str, steps: List[str], procedure_id: str = "user_input") -> DetectionResult:
        procedure = Procedure(id=procedure_id, title=title, steps=steps)
        return self.run_single(procedure)

    def explain_detection(self, result: DetectionResult) -> str:
        return self.detector.generate_explanation(result)

    def _save_analysis(self, results, report):
        self.analyzer.save_results(str(self.output_dir / "analysis_results.json"))
        with open(self.output_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        try:
            self.analyzer.plot_analysis(str(self.output_dir / "plots"))
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    def _save_results(self, output):
        with open(self.output_dir / "detection_results.json", 'w', encoding='utf-8') as f:
            json.dump(output.to_dict(), f, indent=2)
        # Summary CSV
        with open(self.output_dir / "summary.csv", 'w', encoding='utf-8') as f:
            f.write("procedure_id,title,num_steps,num_gaps,avg_confidence\n")
            for r in output.detection_results:
                f.write(f"{r.procedure_id},{r.procedure_title},{len(r.original_steps)},{r.num_gaps_detected},{r.avg_confidence:.3f}\n")
        logger.info(f"Results saved to {self.output_dir}")

    def _generate_report(self, output) -> str:
        lines = [
            "=" * 70, "MISSING STEP DETECTION - PIPELINE REPORT", "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
            "SUMMARY", "-" * 40,
            f"Total procedures: {output.total_procedures}",
            f"Total gaps detected: {output.total_gaps_detected}",
            f"Essential gaps: {output.total_essential_gaps}",
            f"Avg gaps per procedure: {output.avg_gaps_per_procedure:.2f}",
            f"Avg confidence: {output.avg_confidence:.2%}", "",
            "TIMING", "-" * 40,
            f"Total time: {output.total_time:.2f}s",
            f"Analysis time: {output.analysis_time:.2f}s",
            f"Detection time: {output.detection_time:.2f}s", "",
        ]
        if output.evaluation_metrics:
            lines.extend([
                "EVALUATION", "-" * 40,
                f"Precision: {output.evaluation_metrics['precision']:.3f}",
                f"Recall: {output.evaluation_metrics['recall']:.3f}",
                f"F1 Score: {output.evaluation_metrics['f1']:.3f}", "",
            ])
        lines.extend(["=" * 70])
        report = "\n".join(lines)
        with open(self.output_dir / "pipeline_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        return report


def run_demo():
    """Run demonstration."""
    print("\n" + "=" * 70)
    print("MISSING STEP DETECTION SYSTEM - DEMO")
    print("=" * 70)

    config = PipelineConfig(use_synthetic_data=True, max_samples=20, verbose=True)
    pipeline = InferencePipeline(config)
    output = pipeline.run(data_source="synthetic")

    # Demo single procedure
    print("\n" + "=" * 70)
    print("SINGLE PROCEDURE DEMO")
    demo_proc = Procedure(
        id="demo", title="How to Make Tea",
        steps=["Fill kettle with water.", "Pour hot water into cup with tea bag.",
               "Wait 3-5 minutes.", "Add milk if desired."]
    )
    result = pipeline.run_single(demo_proc)
    print(pipeline.explain_detection(result))
    return output


if __name__ == "__main__":
    run_demo()
