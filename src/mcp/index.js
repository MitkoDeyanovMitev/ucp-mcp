#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { execFileSync } from "child_process";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..", "..");
const baseDataDir = path.join(rootDir, "embeddings");
const runDir = path.join(rootDir, "run-data");

// Non-destructive data bootstrapping isolating active client reads
if (!fs.existsSync(runDir) && fs.existsSync(baseDataDir)) {
  console.error(
    "Bootstrapping isolated runtime DB workspace from tracked base structures...",
  );
  try {
    fs.cpSync(baseDataDir, runDir, { recursive: true });
  } catch (err) {}
}

const server = new Server(
  { name: "ucp-mcp-delegated-server", version: "1.0.0" },
  { capabilities: { tools: {}, prompts: {} } },
);

// 1. Prompts Handler
server.onRequest(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "migration-and-integration-guide",
        description:
          "System instructions for processing cross-repository updates and feature modules.",
      },
    ],
  };
});

server.onRequest(GetPromptRequestSchema, async (request) => {
  if (request.params.name !== "migration-and-integration-guide") {
    throw new Error("Requested prompt target not recognized.");
  }
  const skillsPath = path.join(__dirname, "skills.md");
  const content = fs.existsSync(skillsPath)
    ? fs.readFileSync(skillsPath, "utf-8")
    : "Skill documentation not located.";
  return {
    description:
      "System guidelines governing codebase updates and cross-repo features.",
    messages: [
      {
        role: "user",
        content: { type: "text", text: content },
      },
    ],
  };
});

// 2. Tools Schema Definitions
server.onRequest(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "query_ucp_context",
        description:
          "Searches local persistent multi-repository distinct context tables in parallel delegating semantic matching engine.",
        inputSchema: {
          type: "object",
          properties: {
            query_text: {
              type: "string",
              description:
                "Natural language text description of target features, protocols, or multi-repo components.",
            },
            model: {
              type: "string",
              enum: ["gemma", "nomic"],
              description:
                "Embedding model target layer to search against. nomic leverages an expansive code-syntax optimized context.",
            },
          },
          required: ["query_text"],
        },
      },
    ],
  };
});

// 3. Execution Router delegating directly to Python searching sidecar
server.onRequest(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "query_ucp_context") {
    throw new Error("Unknown tool target requested.");
  }

  const { query_text, model = "nomic" } = request.params.arguments;

  try {
    const sidecarPath = path.resolve(
      rootDir,
      "src",
      "generator",
      "lance_search_client.py",
    );
    const pythonInterpreter = path.resolve(rootDir, ".venv", "bin", "python");

    if (!fs.existsSync(sidecarPath) || !fs.existsSync(pythonInterpreter)) {
      return {
        content: [
          {
            type: "text",
            text: "Universal searching sidecar logic script or Python interpreter not found.",
          },
        ],
      };
    }

    // Safely invoke Python sidecar mapping multi-table layouts natively
    const stdout = execFileSync(
      pythonInterpreter,
      [sidecarPath, "--model", model, "--query", query_text],
      { encoding: "utf-8", stdio: ["pipe", "pipe", "ignore"] },
    );

    const parsed = JSON.parse(stdout.trim());
    if (parsed.error) {
      return {
        content: [
          {
            type: "text",
            text: `Database lookup trace reported: ${parsed.error}`,
          },
        ],
      };
    }

    const results = parsed.results || [];
    if (results.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "No relevant matching context blocks retrieved across distinct repository tables.",
          },
        ],
      };
    }

    const formattedText = results
      .map((r) => {
        const repo = r.repo_name
          ? `[Repo: ${r.repo_name}]`
          : `[Table: ${r.table}]`;
        const file = r.file_path ? `[File: ${r.file_path}]` : "";
        const dist =
          r._distance !== undefined
            ? ` [Distance: ${Number(r._distance).toFixed(3)}]`
            : "";
        return `${repo}${file}${dist}\n${r.text || JSON.stringify(r)}`;
      })
      .join("\n\n---\n\n");

    return { content: [{ type: "text", text: formattedText }] };
  } catch (err) {
    return {
      content: [
        {
          type: "text",
          text: `Error executing parallel cross-table lookup delegation: ${err.message}`,
        },
      ],
    };
  }
});

// Verify native host environment dependencies prior to activating server loops
async function verifyHostDependencies() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    const resp = await fetch("http://localhost:11434/api/tags", {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (resp.ok) {
      const data = await resp.json();
      const models = (data.models || []).map((m) => m.name);
      const hasNomic = models.some((m) => m.includes("nomic-embed-text"));
      const hasGemma = models.some((m) => m.includes("embeddinggemma"));

      if (!hasNomic) {
        console.error(
          "\n⚠️ UCP MCP Server Notice: Recommended model 'nomic-embed-text' missing from active host engine cache. Pulling automatically or manually via 'ollama pull nomic-embed-text' recommended.\n",
        );
      }
      if (!hasGemma) {
        console.error(
          "\n⚠️ UCP MCP Server Notice: High-fidelity model 'embeddinggemma' missing from active host engine cache.\n",
        );
      }
    }
  } catch (err) {
    console.error(
      "\n⚠️ UCP MCP Server Startup Alert: Standalone background Ollama service on port 11434 is currently unreachable. Automatic client string vectorization requests will drop until 'ollama serve' is launched natively.\n",
    );
  }
}

await verifyHostDependencies();

const transport = new StdioServerTransport();
await server.connect(transport);
console.error(
  "Universal Context Protocol (ucp-mcp) Delegated Server connected successfully.",
);
