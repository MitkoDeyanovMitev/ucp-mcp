#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# Guarantee the project root directory is accessible in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.embeddings_generator.gemma_encoder import GemmaGenerator
from src.embeddings_generator.nomic_encoder import NomicGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Modular embedding generation launcher"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic",
        choices=["gemma", "nomic"],
        help="Target embedding generator module to trigger",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Process only files modified since the previous repository synchronization pass",
    )
    args = parser.parse_args()

    if args.model == "gemma":
        generator = GemmaGenerator()
    else:
        generator = NomicGenerator()

    generator.run(incremental=args.incremental)


if __name__ == "__main__":
    main()
