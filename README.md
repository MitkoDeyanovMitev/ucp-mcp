# 🚀 Universal Context Protocol MCP Server (`ucp-mcp`)

A highly responsive, zero-infrastructure Model Context Protocol (MCP) server leveraging serverless **LanceDB** vector storage. Distributed natively via `npx`, this server provides instant-start semantic code and document retrieval over universal knowledge bases directly within IDE assistants.

To support deep enterprise-grade data processing, `ucp-mcp` deploys a **Hybrid (Polyglot) Architecture**:
1. **Ultra-Fast Server Runtime (`src/mcp/index.js`):** Built in pure Node.js/ES Modules. Provides instant startup (~50ms) and dynamic text-to-vector resolution bridging without heavy database bindings.
2. **Object-Oriented Generator Suite (`src/generator/`):** Built in Python. Uses an extensible class hierarchy (`BaseEmbeddingGenerator` in `base_encoder.py`) encapsulating multi-repository shallow cloning, regex-driven boilerplate elimination, task instruction prefixing, and highly concentrated table optimization.

---

## 💾 Persistent Storage Layout

To guarantee optimal retrieval speed and isolated scalability, vector tables are decoupled hierarchically by model layer and individual repository inside the persistent base folder:

```text
./embeddings/
  ├── nomic/                        # Code-Syntax Optimized Database Root (768-Dim)
  │    ├── ucp.lance/               # Highly concentrated table for core protocol schemas
  │    ├── python-sdk.lance/        # Table for Python SDK frameworks
  │    └── js-sdk.lance/            # Table for TypeScript/Zod schemas
  └── gemma/                        # Deep Conceptual Database Root (768-Dim Native)
       └── ...
```

Upon server initialization, non-destructive bootstrap logic automatically duplicates tracked tables from `./embeddings` into an ignored runtime directory (`./run-data`), perfectly isolating active IDE read operations from persistent Git state.

---

## 🛠️ Public Prerequisites & Runtime Grace

### **For End-Users (IDE Client Consumption)**
To execute searches client-side natively using free-text query strings, the MCP Server sidecar formats instructions and requests vector arrays via a local host engine. 

> [!IMPORTANT]
> **Client-Side Dependency**: End-users consuming this server via their chat extensions must have the independent **Ollama** background service installed and running on port `11434` with the corresponding model weights pulled:
> ```bash
> # Pull target weights locally before executing queries inside your IDE
> ollama pull nomic-embed-text
> ollama pull embeddinggemma
> ```

#### **Graceful Troubleshooting**
If the local service daemon is missing or unreachable, the server **does not crash**. Instead, the search execution bridge catches connection timeouts and safely bubbles highly instructive help messages directly back into the developer's IDE chat window:
```text
Database lookup trace reported: Failed to generate vector via local Ollama service: Connection refused. Please ensure Ollama is installed and running locally.
```

---

## 📥 Multi-Repository Ingestion Pipeline

External codebases are managed declaratively via `sources.json`:
```json
{
  "repositories": [
    {
      "name": "ucp",
      "type": "git",
      "url": "https://github.com/Universal-Commerce-Protocol/ucp.git",
      "branch": "gh-pages",
      "enabled": true
    }
  ]
}
```

### **Running Database Rebuilds (Maintainer Setup)**
Prepare your virtual environment inside Python 3.10+:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Execute parallel multi-threaded batch builds triggering your desired model subclass layer:
```bash
# Rebuild highly concentrated Nomic repository matrices stably (max_workers=2)
python3 src/generator/build_vector_databases.py --model nomic

# Rebuild high-fidelity Gemma documentation layers
python3 src/generator/build_vector_databases.py --model gemma
```

---

## 🔌 IDE Integration (Dual-Environment Entrypoints)

Configure your preferred chat client extension using your desired server interface layer:

### **Option A: Node.js Standard I/O Bridge (Zero-Install via `npx`)**
Provides absolute frictionless distribution natively over GitHub without requiring local add-on compilation:
* **Command:** `npx`
* **Arguments:** `["-y", "github:your-username/ucp-mcp"]`

### **Option B: Native Python Entrypoint (`FastMCP`)**
Executes the standard server protocol pure Python engine natively:
* **Command:** `/absolute/path/to/ucp-mcp/.venv/bin/python3`
* **Arguments:** `["/absolute/path/to/ucp-mcp/src/mcp/server.py"]`

---

## 🎯 Exposed MCP Capabilities

* **Prompts:** Discovers `migration-and-integration-guide` dynamically injecting integration instruction sets sourced from `src/mcp/skills.md`.
* **Tools:** Exposes `query_ucp_context` accepting natural language search strings (`query_text`) and target model layers (`model: "nomic" | "gemma"`). At runtime, queries compute exact spatial coordinates via local engine execution to return globally ranked nearest-neighbor documentation chunks.
