#!/bin/bash
# Resolve the absolute directory of this script's folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
MODEL="${UCP_MCP_MODEL:-nomic}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
SKIP_PULL=false
SYNC_MODE="incremental"

show_help() {
  echo "Usage: run-server.sh [options] [fastmcp_options]"
  echo ""
  echo "Options:"
  echo "  -m <nomic|gemma>           Target embedding model to load (default: nomic)"
  echo "  -u <url>                   Ollama API host URL (default: http://localhost:11434)"
  echo "  -s <full|incremental|skip> Database sync mode at startup (default: incremental)"
  echo "  -n                         Skip verifying/pulling the Ollama model automatically"
  echo "  -h                         Display this help message"
  echo ""
}

# Parse options using POSIX standard getopts
while getopts "m:u:s:nh" opt; do
  case "$opt" in
    m)
      MODEL="$OPTARG"
      ;;
    u)
      OLLAMA_HOST="$OPTARG"
      ;;
    s)
      SYNC_MODE="$OPTARG"
      ;;
    n)
      SKIP_PULL=true
      ;;
    h)
      show_help
      exit 0
      ;;
    \?)
      show_help >&2
      exit 1
      ;;
  esac
done

# Shift positional parameters so that $@ only holds unhandled arguments for FastMCP
shift $((OPTIND-1))

# Map UCP model names to Ollama model names and validate the choice
case "$MODEL" in
  gemma)
    OLLAMA_MODEL="embeddinggemma"
    ;;
  nomic)
    OLLAMA_MODEL="nomic-embed-text"
    ;;
  *)
    echo "Error: Invalid model choice '$MODEL'. Must be 'nomic' or 'gemma'." >&2
    exit 1
    ;;
esac

# Export environment variables so the Python server and subprocesses inherit them
export OLLAMA_HOST
export UCP_MCP_MODEL="$MODEL"

# Validate sync mode choice
if [[ "$SYNC_MODE" != "full" && "$SYNC_MODE" != "incremental" && "$SYNC_MODE" != "skip" ]]; then
  echo "Error: Invalid sync mode '$SYNC_MODE'. Must be 'full', 'incremental', or 'skip'." >&2
  exit 1
fi

# Check if embeddings for the selected model are present. If missing and not rebuilding, download them.
if [[ "$SYNC_MODE" != "full" && ! -d "$DIR/embeddings/$MODEL" ]]; then
  BRANCH_SUFFIX="$MODEL"
  echo "📥 UCP database files for model '$MODEL' not found locally. Downloading pre-compiled assets from branch 'release-${BRANCH_SUFFIX}'..." >&2
  
  # Download the zip archive of the model's release branch
  ZIP_URL="https://github.com/MitkoDeyanovMitev/ucp-mcp/archive/refs/heads/release-${BRANCH_SUFFIX}.zip"
  curl -sL "$ZIP_URL" -o "$DIR/release.zip"
  
  if [ -f "$DIR/release.zip" ]; then
    echo "Extracting database assets..." >&2
    # Unzip only the specific model embeddings folder from the archive to a temp folder
    unzip -q "$DIR/release.zip" "ucp-mcp-release-${BRANCH_SUFFIX}/embeddings/${MODEL}/*" -d "$DIR/temp_extract" > /dev/null
    
    if [ -d "$DIR/temp_extract/ucp-mcp-release-${BRANCH_SUFFIX}/embeddings/${MODEL}" ]; then
      mkdir -p "$DIR/embeddings"
      mv "$DIR/temp_extract/ucp-mcp-release-${BRANCH_SUFFIX}/embeddings/${MODEL}" "$DIR/embeddings/${MODEL}"
      echo "✅ Databases for model '$MODEL' loaded successfully!" >&2
    else
      echo "⚠️ Error: Failed to extract databases from release package." >&2
    fi
    
    # Clean up temporary files
    rm -rf "$DIR/release.zip" "$DIR/temp_extract"
  else
    echo "⚠️ Error: Failed to download database release package." >&2
  fi
fi

# Configure database synchronization behavior at server startup
if [[ "$SYNC_MODE" == "incremental" ]]; then
  export UCP_MCP_SKIP_SYNC="false"
else
  export UCP_MCP_SKIP_SYNC="true"
fi

if [[ "$SKIP_PULL" == "false" ]]; then
  # If Ollama is offline, check if we can start it automatically
  if ! curl -fsSL "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    OS_TYPE="$(uname -s)"
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
      if [ -d "/Applications/Ollama.app" ] || [ -d "$HOME/Applications/Ollama.app" ]; then
        echo "Ollama App detected offline. Starting background service..." >&2
        open -a Ollama >/dev/null 2>&1
      fi
    elif [[ "$OS_TYPE" == "Linux" || "$OS_TYPE" == "MINGW"* || "$OS_TYPE" == "CYGWIN"* || "$OS_TYPE" == "MSYS"* ]]; then
      if command -v ollama >/dev/null 2>&1; then
        echo "Starting Ollama background service via CLI..." >&2
        ollama serve >/dev/null 2>&1 &
      fi
    fi
    
    # Wait up to 5 seconds for Ollama to bind to the port
    for i in {1..5}; do
      if curl -fsSL "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi

  # Check if the Ollama server is running now
  if ! curl -fsSL "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    echo -e "\n⚠️  UCP MCP Server Warning: Ollama service at '${OLLAMA_HOST}' is unreachable." >&2
    if ! command -v ollama >/dev/null 2>&1; then
      echo -e "Ollama is missing. You can install it natively via: brew install --cask ollama" >&2
    else
      echo -e "Please ensure Ollama is running ('ollama serve'), otherwise semantic search queries will fail.\n" >&2
    fi
  else
    # Check if the required model is pulled
    if ! curl -s "${OLLAMA_HOST}/api/tags" | grep -q "$OLLAMA_MODEL"; then
      echo -e "\n⚠️  UCP MCP Server Notice: Model '$OLLAMA_MODEL' is missing from Ollama." >&2
      echo -e "Pulling it automatically now..." >&2
      ollama pull "$OLLAMA_MODEL" >&2
    fi
  fi
fi

# Conditionally trigger a FULL database rebuild inside shell before server startup
if [[ "$SYNC_MODE" == "full" ]]; then
  echo "Starting FULL database rebuild for model '$MODEL'..." >&2
  "$DIR/.venv/bin/python" "$DIR/src/embeddings_generator/build_vector_databases.py" --model "$MODEL"
  if [[ $? -ne 0 ]]; then
    echo "Error: Database rebuild failed. Server launch aborted." >&2
    exit 1
  fi
fi

# Run the server using relative virtualenv Python execution, passing down unhandled CLI commands (e.g. dev, run)
exec "$DIR/.venv/bin/python" "$DIR/src/mcp/server.py" "$@"
