#!/usr/bin/env python3
import sys
import os
from contextlib import redirect_stdout
import lancedb
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.embeddings_generator.gemma_encoder import GemmaGenerator
from src.embeddings_generator.nomic_encoder import NomicGenerator

mcp = FastMCP("ucp-mcp-python-server")


@mcp.prompt("migration-and-integration-guide")
def migration_guide() -> str:
    """System instructions for processing cross-repository updates and feature modules."""
    skills_path = os.path.join(os.path.dirname(__file__), "skills.md")
    if os.path.exists(skills_path):
        with open(skills_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Skill documentation guidelines not located."


default_model = os.environ.get("UCP_MCP_MODEL", "nomic")


@mcp.tool("query_ucp_context")
def query_ucp_context(query_text: str, model: str = default_model) -> str:
    """Searches local persistent multi-repository distinct context tables natively in Python using semantic similarity math.

    Args:
        query_text: Natural language text description of target features, protocols, or multi-repo components.
        model: Embedding model target layer to search against. nomic leverages high-capacity syntax encoding.
    """
    base_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_dir = os.path.join(base_root, "embeddings", model)
    if not os.path.exists(db_dir):
        db_dir = os.path.join(base_root, "run-data", model)
        if not os.path.exists(db_dir):
            return f"Database storage root '{db_dir}' not found. Please populate repository tables via generator scripts."

    try:
        db = lancedb.connect(db_dir)
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
            return f"No repository tables populated inside model layer '{model}'."

        # Searching evaluates across all repository tables concurrently

        if model == "gemma":
            gen = GemmaGenerator()
        else:
            gen = NomicGenerator()

        vec = gen.generate_vector(query_text, is_query=True)

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
        top_results = results[:10]

        if not top_results:
            return "No relevant matching context blocks retrieved across targeted repository tables."

        formatted = []
        for r in top_results:
            repo = (
                f"[Repo: {r['repo_name']}]"
                if r.get("repo_name")
                else f"[Table: {r['table']}]"
            )
            file_p = f"[File: {r['file_path']}]" if r.get("file_path") else ""
            dist = f" [Distance: {r['_distance']:.3f}]" if "_distance" in r else ""
            formatted.append(f"{repo}{file_p}{dist}\n{r['text']}")

        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error executing native Python cross-table semantic context lookup: {str(e)}"


def run_startup_sync(model: str, skip_sync: bool):
    if skip_sync:
        print("Skipping startup embedding synchronization pass.", file=sys.stderr)
        return

    try:
        if model == "gemma":
            generator = GemmaGenerator()
        else:
            generator = NomicGenerator()

        # Redirect all stdout to stderr to prevent corrupting MCP stdio stream
        with redirect_stdout(sys.stderr):
            generator.run(incremental=True)

    except Exception as e:
        print(
            f"⚠️ Failed to run incremental embedding generator: {e}",
            file=sys.stderr,
        )


def main():
    model = os.environ.get("UCP_MCP_MODEL", "nomic")
    skip_sync = os.environ.get("UCP_MCP_SKIP_SYNC") == "true"

    if "--model" in sys.argv:
        try:
            idx = sys.argv.index("--model")
            if idx + 1 < len(sys.argv):
                model = sys.argv[idx + 1]
                sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
        except ValueError:
            pass

    if "--skip-sync" in sys.argv:
        skip_sync = True
        try:
            sys.argv.remove("--skip-sync")
        except ValueError:
            pass

    run_startup_sync(model, skip_sync)
    mcp.run()


if __name__ == "__main__":
    main()
