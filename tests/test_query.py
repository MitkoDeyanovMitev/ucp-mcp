import sys
import os

# Guarantee project root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp.server import query_ucp_context


def main():
    # Parse command line arguments
    query = "core shopping capabilities checkout cart order"
    model = "nomic"

    if len(sys.argv) > 1:
        query = sys.argv[1]
    if len(sys.argv) > 2:
        model = sys.argv[2]

    print(f"Running query: '{query}' using model '{model}'...")
    try:
        result = query_ucp_context(query, model)
        print("\n--- Search Results ---")
        print(result)
    except Exception as e:
        print(f"Error executing search: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
