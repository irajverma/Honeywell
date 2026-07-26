"""
generate_data.py — CLI entry point for the synthetic data generator.

Usage:
    python generate_data.py
    python generate_data.py --days 90 --seed 123 --output-dir data/
    python generate_data.py --test-days 15 --drift-frac 0.15
"""

import argparse
import sys
import time

from datagen.config import GeneratorConfig
from datagen.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Anomaly Detection — Synthetic Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=60,
                        help="Total simulation window in days")
    parser.add_argument("--test-days", type=int, default=12,
                        help="Number of days reserved for the test split")
    parser.add_argument("--drift-frac", type=float, default=0.12,
                        help="Fraction of entities with concept drift in test period")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--users", type=int, default=50,
                        help="Number of user entities")
    parser.add_argument("--service-accounts", type=int, default=10,
                        help="Number of service account entities")
    parser.add_argument("--edge-devices", type=int, default=20,
                        help="Number of edge device entities")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for generated CSVs")

    args = parser.parse_args()

    config = GeneratorConfig(
        num_users=args.users,
        num_service_accounts=args.service_accounts,
        num_edge_devices=args.edge_devices,
        sim_days=args.days,
        test_days=args.test_days,
        concept_drift_frac=args.drift_frac,
        random_seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("Behavioral Anomaly Detection -- Synthetic Data Generator")
    print(f"{'='*60}")
    print(f"Entities: {config.total_entities} "
          f"({config.num_users}U + {config.num_service_accounts}S + "
          f"{config.num_edge_devices}E)")
    print(f"Window:   {config.sim_days} days "
          f"(train: {config.sim_days - config.test_days}d, "
          f"test: {config.test_days}d)")
    print(f"Drift:    {config.concept_drift_frac*100:.0f}% of entities in test period")
    print(f"Seed:     {config.random_seed}")
    print(f"Output:   {args.output_dir}/")
    print(f"{'='*60}\n")

    start_time = time.time()
    events_df, gt_df = generate_dataset(config, output_dir=args.output_dir)
    elapsed = time.time() - start_time

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
