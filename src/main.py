import argparse, yaml, uuid, os, json
from pathlib import Path

from src.train import run
from src.evaluate import evaluate

def load_yaml(path):
    with open(path) as fp:
        return yaml.safe_load(fp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--full-experiment", action="store_true")
    args = parser.parse_args()

    if args.smoke_test == args.full_experiment:
        raise ValueError("Choose exactly one of --smoke-test / --full-experiment")

    cfg_path = "config/smoke_test.yaml" if args.smoke_test else "config/full_experiment.yaml"
    cfg = load_yaml(cfg_path)

    run_id = f"{'smoke' if args.smoke_test else 'full'}_{uuid.uuid4().hex[:6]}"
    
    # Phase 1: Smoke Test
    if args.smoke_test:
        print("--- Running Smoke Test ---")
        json_log = run(cfg, run_id)
        evaluate(json_log, cfg["description"])
        print("--- Smoke Test Passed ---")
        return

    # Phase 2: Full Experiment (can be run directly or after a smoke test pass)
    if args.full_experiment:
        print("--- Running Full Experiment ---")
        json_log = run(cfg, run_id)
        evaluate(json_log, cfg["description"])
        print("--- Full Experiment Finished ---")

if __name__ == "__main__":
    main()
