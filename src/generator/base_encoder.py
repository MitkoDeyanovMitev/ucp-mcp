import os
import json
import glob
import shutil
import subprocess
import time
import re
import pandas as pd
import lancedb
from abc import ABC, abstractmethod


class BaseEmbeddingGenerator(ABC):
    def __init__(self, model_name: str, dimensions: int):
        self.model_name = model_name
        self.dimensions = dimensions
        self.config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "sources.json")
        )
        self.staging_base = ".staging-repos"
        self.db_base = f"./embeddings/{self.model_name}"

    @abstractmethod
    def generate_vector(self, text: str, is_query: bool = False) -> list[float]:
        """Extracts precise numeric vector arrays mapping input semantic strings."""
        pass

    def generate_vectors(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        """Extracts batch multi-dimensional numeric vector arrays mapping input string lists."""
        return [self.generate_vector(t, is_query=is_query) for t in texts]

    @abstractmethod
    def chunk_text(self, content: str, file_path: str) -> list[str]:
        """Extracts language-aware syntax chunks optimized for target encoder constraints."""
        pass

    def _clean_html(self, html_str: str) -> str:
        """Strips visual layout bloat, navigation menus, scripts, and standard web footer boilerplate from raw HTML pages."""
        main_match = re.search(
            r"<main[^>]*>(.*?)</main>", html_str, re.DOTALL | re.IGNORECASE
        )
        if not main_match:
            main_match = re.search(
                r"<article[^>]*>(.*?)</article>", html_str, re.DOTALL | re.IGNORECASE
            )
        if not main_match:
            main_match = re.search(
                r'<div[^>]*class="[^"]*md-content[^"]*"[^>]*>(.*)',
                html_str,
                re.DOTALL | re.IGNORECASE,
            )

        content = main_match.group(1) if main_match else html_str
        content = re.sub(
            r"<(script|style|nav|header|footer)[^>]*>.*?</\1>",
            " ",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r"<svg[^>]*>.*?</svg>", " ", content, flags=re.DOTALL | re.IGNORECASE
        )
        content = re.sub(r"<!--.*?-->", " ", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>", " ", content)
        return re.sub(r"\s+", " ", content).strip()

    def _clean_code_content(self, content: str, file_path: str) -> str:
        """Removes standard licensing boilerplate and copyright headers from code files to improve embedding semantic density."""
        content_stripped = content.lstrip()
        # Handle C-style block comment at the very beginning
        if content_stripped.startswith("/*"):
            end_idx = content_stripped.find("*/")
            if end_idx != -1:
                header = content_stripped[: end_idx + 2]
                if any(
                    kw in header.lower()
                    for kw in [
                        "copyright",
                        "license",
                        "apache",
                        "mit",
                        "warranty",
                        "is distributed on an",
                    ]
                ):
                    content_stripped = content_stripped[end_idx + 2 :].lstrip()

        # Handle contiguous blocks of line comments (# or //) at the beginning of the file
        lines = content_stripped.splitlines()
        header_lines = 0
        has_license_kw = False

        for line in lines:
            sline = line.strip()
            if not sline:
                header_lines += 1
                continue
            if sline.startswith("#") or sline.startswith("//"):
                header_lines += 1
                line_lower = sline.lower()
                if any(
                    kw in line_lower
                    for kw in [
                        "copyright",
                        "license",
                        "apache",
                        "mit",
                        "warranty",
                        "basis,",
                        "is distributed on an",
                    ]
                ):
                    has_license_kw = True
            else:
                break

        if has_license_kw:
            return "\n".join(lines[header_lines:]).lstrip()

        return content_stripped

    def shallow_clone(self, repo_config: dict) -> str:
        name = repo_config["name"]
        url = repo_config["url"]
        branch = repo_config.get("branch", "main")
        target_dir = os.path.join(self.staging_base, name)

        if os.path.exists(target_dir):
            print(f"Removing cached staging tree for {name}...")
            shutil.rmtree(target_dir)

        print(f"Shallow cloning {name} from {url} (branch: {branch})...")
        os.makedirs(self.staging_base, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, url, target_dir],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository {name}: {e}")
        return target_dir

    def get_files_to_index(
        self, base_path: str, includes: list[str], excludes: list[str]
    ) -> list[str]:
        matched_files = set()
        cwd = os.getcwd()
        try:
            os.chdir(base_path)
            for pattern in includes:
                for f in glob.glob(pattern, recursive=True):
                    if os.path.isfile(f):
                        matched_files.add(f)

            for pattern in excludes:
                for f in glob.glob(pattern, recursive=True):
                    if f in matched_files:
                        matched_files.remove(f)
        finally:
            os.chdir(cwd)

        return [os.path.join(base_path, f) for f in matched_files]

    def run(self):
        start_time = time.time()
        print(
            f"Starting highly modular embedding pipeline for model tier: {self.model_name}"
        )
        if not os.path.exists(self.config_path):
            print(f"Configuration manifest {self.config_path} not located.")
            return

        with open(self.config_path, "r") as f:
            config = json.load(f)

        records = []

        for repo in config.get("repositories", []):
            if not repo.get("enabled", True):
                print(f"Skipping disabled repository entry: {repo['name']}")
                continue

            name = repo["name"]
            repo_type = repo.get("type", "local")

            if repo_type == "git":
                base_path = self.shallow_clone(repo)
            else:
                base_path = repo.get("path", ".")

            if not os.path.exists(base_path):
                continue

            includes = repo.get("include", ["**/*"])
            global_excludes = [
                "**/.git/**",
                "**/node_modules/**",
                "**/venv/**",
                "**/.venv/**",
                "**/__pycache__/**",
                "**/dist/**",
                "**/build/**",
                "**/*.lock",
                "**/*.png",
                "**/*.jpg",
                "**/*.ico",
                "**/*.pdf",
            ]
            excludes = repo.get("exclude", []) + global_excludes

            files = self.get_files_to_index(base_path, includes, excludes)
            print(
                f"Discovered {len(files)} matching sources inside target index: {name}"
            )

            repo_chunks = []
            for file_path in files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except Exception:
                    continue

                rel_path = os.path.relpath(file_path, base_path)
                if file_path.lower().endswith(".html"):
                    content = self._clean_html(content)
                    if not content:
                        continue
                else:
                    content = self._clean_code_content(content, rel_path)
                    if not content.strip():
                        continue

                chunks = self.chunk_text(content, rel_path)
                for idx, chunk in enumerate(chunks):
                    meta_text = f"Repo: {name} | File: {rel_path} | Chunk: {idx + 1} | Content: {chunk}"
                    repo_chunks.append((meta_text, rel_path))

            total_chunks = len(repo_chunks)
            if total_chunks == 0:
                print(
                    f"No valid extracted text chunks found for repository '{name}'. Skipping embedding generation.",
                    flush=True,
                )
                continue

            print(
                f"Extracted {total_chunks} semantic chunks from repository '{name}'. Initiating vectorized bulk embedding phase...",
                flush=True,
            )

            batch_size = 32
            for b_start in range(0, total_chunks, batch_size):
                b_chunks = repo_chunks[b_start : b_start + batch_size]
                b_texts = [c[0] for c in b_chunks]

                b_start_time = time.time()
                try:
                    b_vectors = self.generate_vectors(b_texts, is_query=False)
                except Exception as e:
                    print(
                        f"Error processing vector array chunk block [{b_start + 1}:{b_start + len(b_chunks)}]: {e}. Supplying fallback isolation arrays.",
                        flush=True,
                    )
                    b_vectors = [[0.01] * self.dimensions for _ in b_texts]

                b_duration = time.time() - b_start_time
                print(
                    f"Processed vector slice [{b_start + 1} - {b_start + len(b_chunks)} / {total_chunks}] in {b_duration:.2f}s",
                    flush=True,
                )

                for idx, (meta_text, rel_path) in enumerate(b_chunks):
                    records.append(
                        {
                            "vector": b_vectors[idx],
                            "text": meta_text,
                            "file_path": rel_path,
                            "repo_name": name,
                        }
                    )

        if not records:
            print("No source mapping records generated. Pipeline execution complete.")
            return

        os.makedirs(self.db_base, exist_ok=True)
        print(f"Connecting to connected database root: {self.db_base}...")
        db = lancedb.connect(self.db_base)

        df_all = pd.DataFrame(records)

        for repo_name, group_df in df_all.groupby("repo_name"):
            base_table_name = "".join(
                [c if c.isalnum() or c in "-_" else "_" for c in repo_name]
            )

            # Enforce strict row partitioning to guarantee single generated binary table fragments stay safely below GitHub's 100MB file thresholds
            max_rows_per_table = 2000
            total_repo_rows = len(group_df)

            if total_repo_rows <= max_rows_per_table:
                print(
                    f"Committing distinct isolated table partition: {base_table_name}...",
                    flush=True,
                )
                table = db.create_table(
                    base_table_name, data=group_df, mode="overwrite"
                )
                try:
                    table.optimize()
                except Exception:
                    pass
            else:
                print(
                    f"Repository '{repo_name}' contains {total_repo_rows} rows. Partitioning into multi-table fragments to enforce <100MB Git push limits...",
                    flush=True,
                )
                for p_idx, start_r in enumerate(
                    range(0, total_repo_rows, max_rows_per_table)
                ):
                    sub_df = group_df.iloc[start_r : start_r + max_rows_per_table]
                    sub_table_name = f"{base_table_name}_part{p_idx + 1}"
                    print(
                        f" - Committing fragmented sub-table: {sub_table_name} ({len(sub_df)} rows)...",
                        flush=True,
                    )
                    table = db.create_table(
                        sub_table_name, data=sub_df, mode="overwrite"
                    )
                    try:
                        table.optimize()
                    except Exception:
                        pass

        duration = time.time() - start_time
        print(
            f"Pipeline processing complete! Populated {len(df_all['repo_name'].unique())} distinct repository tables under model layer '{self.model_name}' in {duration:.2f} seconds."
        )
