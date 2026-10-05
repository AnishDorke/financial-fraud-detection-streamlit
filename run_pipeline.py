"""
End-to-End Orchestrator for Financial Fraud Detection Pipeline
Executes schema validation, boundary checks, model loading, and parity verification.
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def run_step(step_name: str, command: list[str]) -> bool:
    print(f"\n{'='*70}\n[RUNNING] {step_name}\n{'='*70}")
    result = subprocess.run(command, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"\n❌ [FAILED] {step_name} exited with return code {result.returncode}")
        return False
    print(f"✔ [PASSED] {step_name}")
    return True

def main():
    steps = [
        ("Step 1: Raw Data Integrity & Schema Validation", [sys.executable, "src/validate_data.py"]),
        ("Step 2: Historical Feature Temporal Boundary Verification", [sys.executable, "src/verify_history_boundaries.py"]),
        ("Step 3: LightGBM Model Loading Check", [sys.executable, "-c", "from src.inference_pipeline import FraudInferencePipeline; p=FraudInferencePipeline(); print('MODEL LOAD OK')"]),
        ("Step 4: Inference Pipeline Parity Verification", [sys.executable, "src/verify_prediction_parity.py"]),
        ("Step 5: Alert Review Capacity Threshold Evaluation", [sys.executable, "src/evaluate_alert_capacity.py"]),
    ]

    print("Starting Fraud Detection End-to-End Pipeline Audit...")
    for name, cmd in steps:
        if not run_step(name, cmd):
            sys.exit(1)

    print(f"\n{'='*70}\n🎉 ALL PIPELINE AUDIT GATES PASSED SUCCESSFULLY\n{'='*70}\n")

if __name__ == "__main__":
    main()