#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import argparse
import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.embeddings_generator.gemma_encoder import GemmaGenerator
from src.embeddings_generator.nomic_encoder import NomicGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Universal search utility sidecar")
    parser.add_argument("--model", type=str, default="nomic")
    parser.add_argument("--query", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base_root = Path(__file__).resolve().parents[2]
    db_dir = base_root / "embeddings" / args.model
    if not db_dir.exists():
        db_dir = base_root / "run-data" / args.model
        if not db_dir.exists():
            print(
                json.dumps(
                    {"error": f"Database storage not found for model tier {args.model}"}
                )
            )
            return

    try:
        db = lancedb.connect(str(db_dir))
        raw_tables = getattr(
            db, "list_tables", getattr(db, "table_names", lambda: [])
        )()
        tables = []

        if isinstance(raw_tables, dict):
            tables = raw_tables.get("tables", [])
        elif isinstance(raw_tables, (list, tuple)):
            for item in raw_tables:
                if isinstance(item, str):
                    tables.append(item)
                elif (
                    isinstance(item, (list, tuple))
                    and len(item) > 1
                    and isinstance(item[1], list)
                ):
                    tables.extend(item[1])
        else:
            if hasattr(raw_tables, "tables"):
                tables = getattr(raw_tables, "tables")
            else:
                tables = list(raw_tables)

        tables = [t for t in tables if isinstance(t, str) and t != "tables"]

        if not tables:
            print(
                json.dumps({"error": f"No repository tables populated in {args.model}"})
            )
            return

        # Searching evaluates across the entire catalog tables globally

        if args.model == "gemma":
            gen = GemmaGenerator()
        else:
            gen = NomicGenerator()

        vec = gen.generate_vector(args.query, is_query=True)

        results = []
        for t_name in tables:
            try:
                table = db.open_table(t_name)
                df_res = table.search(vec).limit(5).to_pandas()
                for _, row in df_res.iterrows():
                    dist = row.get("_distance", 1.0)
                    results.append(
                        {
                            "repo_name": row.get("repo_name", t_name),
                            "file_path": row.get("file_path", ""),
                            "text": row.get("text", ""),
                            "table": t_name,
                            "_distance": float(dist),
                        }
                    )
            except Exception:
                pass

        results.sort(key=lambda r: r["_distance"])

        print(json.dumps({"results": results[:10]}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
