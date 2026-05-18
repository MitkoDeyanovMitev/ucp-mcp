#!/usr/bin/env python3
import os
import json
import subprocess
import time


def run_query(model: str, query: str):
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    client_script = os.path.join(
        base_root, "src", "embeddings_generator", "lance_search_client.py"
    )
    python_bin = os.path.join(base_root, ".venv", "bin", "python3")

    start_t = time.time()
    try:
        res = subprocess.run(
            [python_bin, client_script, "--model", model, "--query", query],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = time.time() - start_t

        # Parse JSON block out of mixed client outputs
        out_lines = res.stdout.splitlines()
        json_str = ""
        for line in out_lines:
            if line.strip().startswith("{") and "results" in line:
                json_str = line.strip()
                break
        if not json_str and out_lines:
            json_str = out_lines[-1]

        data = json.loads(json_str)
        return data.get("results", []), duration
    except Exception:
        return [], 0.0


def get_folder_size(folder_path: str) -> float:
    total_size = 0
    if not os.path.exists(folder_path):
        return 0.0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # MB


def main():
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    queries_file = os.path.join(base_root, "tests", "evaluation_queries.json")

    with open(queries_file, "r") as f:
        queries = json.load(f)

    models = ["gemma", "nomic"]
    report = {"storage_sizes_mb": {}, "evaluations": []}

    print("Evaluating embedding database footprints...")
    for m in models:
        sz = get_folder_size(os.path.join(base_root, "embeddings", m))
        report["storage_sizes_mb"][m] = round(sz, 2)
        print(f" - {m}: {sz:.2f} MB")

    print("\nExecuting cross-model evaluation benchmark suite...")
    for q in queries:
        q_text = q["query"]
        print(f"\nQuery: '{q_text}'")
        q_res = {"id": q["id"], "query": q_text, "results": {}}
        for m in models:
            chunks, dur = run_query(m, q_text)
            top_chunk = chunks[0] if chunks else None
            q_res["results"][m] = {
                "latency_s": round(dur, 2),
                "chunks_returned": len(chunks),
                "top_distance": round(top_chunk["_distance"], 3)
                if top_chunk and "_distance" in top_chunk
                else None,
                "top_repo": top_chunk["repo_name"] if top_chunk else None,
                "top_file": top_chunk["file_path"] if top_chunk else None,
                "snippet": top_chunk["text"][:200].replace("\n", " ")
                if top_chunk
                else "No results",
            }
            print(
                f" - {m} -> {len(chunks)} chunks in {dur:.2f}s (Top dist: {q_res['results'][m]['top_distance']})"
            )
        report["evaluations"].append(q_res)

    out_report = os.path.join(base_root, "tests", "benchmark_results.json")
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark complete! Results compiled to {out_report}")


if __name__ == "__main__":
    main()
