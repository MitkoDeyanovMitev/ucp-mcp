#!/usr/bin/env python3
import sys
import os
import argparse

# Guarantee the project root directory is accessible in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.generator.gemma_encoder import GemmaGenerator
from src.generator.nomic_encoder import NomicGenerator


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
    args = parser.parse_args()

    if args.model == "gemma":
        generator = GemmaGenerator()
    else:
        generator = NomicGenerator()

    generator.run()


if __name__ == "__main__":
    main()
