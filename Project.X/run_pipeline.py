"""
Orchestration script: End-to-end AI layer setup and testing.
Runs: dataset_builder → trainer → evaluator → main.py with predictor.
"""

import sys
import os
import subprocess

# Paths
PROJECT_ROOT = r'C:\Users\bkiyo\Desktop\Project.X'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

AMQSS_CSV = os.path.join(RESULTS_DIR, 'market_scores.csv')
PRICE_CSV = r'C:\Users\bkiyo\Downloads\EURUSD_60_2025-01-20_2026-01-19.csv'
DATASET_CSV = os.path.join(DATA_DIR, 'dataset.csv')


def run_step(step_name, script_path, args=None):
    """Run a Python script and report status."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            print(f"✓ {step_name} completed successfully")
            return True
        else:
            print(f"✗ {step_name} failed")
            return False
    except Exception as e:
        print(f"✗ {step_name} error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("AMQSS AI LAYER - END-TO-END SETUP")
    print("="*60)
    
    # Step 1: Build dataset
    print("\n[1/4] Building dataset from AMQSS results + price data...")
    if not os.path.exists(AMQSS_CSV):
        print(f"⚠️  No AMQSS results found at {AMQSS_CSV}")
        print("    Run main.py first to generate market_scores.csv")
        print("    Skipping dataset builder for now.")
        dataset_ok = False
    else:
        dataset_builder_script = os.path.join(PROJECT_ROOT, 'dataset_builder.py')
        # Modify dataset_builder to use paths
        dataset_ok = run_step("Dataset Builder", dataset_builder_script)
    
    if not dataset_ok:
        print("\n⚠️  Skipping training without dataset. Run main.py first.")
        return
    
    # Step 2: Train model
    print("\n[2/4] Training model (time-based split, no shuffle)...")
    trainer_script = os.path.join(PROJECT_ROOT, 'trainer.py')
    trainer_ok = run_step("Trainer", trainer_script)
    
    if not trainer_ok:
        print("\n✗ Training failed.")
        return
    
    # Step 3: Evaluate model
    print("\n[3/4] Evaluating model on test set...")
    evaluator_script = os.path.join(PROJECT_ROOT, 'evaluator.py')
    eval_ok = run_step("Evaluator", evaluator_script)
    
    if not eval_ok:
        print("\n✗ Evaluation failed.")
        return
    
    # Step 4: Run main.py with predictor
    print("\n[4/4] Running main.py with new AI predictor...")
    main_script = os.path.join(PROJECT_ROOT, 'main.py')
    main_ok = run_step("Main (with AI)", main_script)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if all([dataset_ok, trainer_ok, eval_ok, main_ok]):
        print("✓ All steps completed successfully!")
        print(f"\nArtifacts:")
        print(f"  Model & Metadata:  {MODELS_DIR}")
        print(f"  Evaluation plots:  {MODELS_DIR}")
        print(f"  Results:           {RESULTS_DIR}")
    else:
        print("✗ Some steps failed. Check output above.")


if __name__ == '__main__':
    main()
