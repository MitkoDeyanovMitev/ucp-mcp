#!/bin/bash
# Resolve the absolute directory of this script's folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 1. Verify local Ollama server is running
echo "Checking local Ollama service status..."
if ! curl -fsSL http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "Error: Local Ollama service is unreachable. Ollama is required to compile embeddings." >&2
  exit 1
fi



# 3. Cache the original branch name to return to it later
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# 4. Compile and publish Nomic database assets
echo "Starting clean database build for Nomic..."
"$DIR/.venv/bin/python" "$DIR/src/embeddings_generator/build_vector_databases.py" --model nomic
if [[ $? -eq 0 ]]; then
  echo "Switching to temporary branch 'release-nomic'..."
  git checkout -B release-nomic
  echo "Staging Nomic database assets..."
  git add -f embeddings/nomic/
  git add src/embeddings_generator/sources.json tests/benchmark_results.json
  git commit -m "data: release nomic database assets"
  git push origin release-nomic --force
  git checkout "$CURRENT_BRANCH"
else
  echo "Error: Nomic database build failed. Release aborted." >&2
  exit 1
fi

# 5. Compile and publish Gemma database assets
echo "Starting clean database build for Gemma..."
"$DIR/.venv/bin/python" "$DIR/src/embeddings_generator/build_vector_databases.py" --model gemma
if [[ $? -eq 0 ]]; then
  echo "Switching to temporary branch 'release-gemma'..."
  git checkout -B release-gemma
  echo "Staging Gemma database assets..."
  git add -f embeddings/gemma/
  git add src/embeddings_generator/sources.json tests/benchmark_results.json
  git commit -m "data: release gemma database assets"
  git push origin release-gemma --force
  git checkout "$CURRENT_BRANCH"
else
  echo "Error: Gemma database build failed. Release aborted." >&2
  exit 1
fi

echo "✅ Database releases successfully compiled and published to branches 'release-nomic' and 'release-gemma'!"
