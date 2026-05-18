import os
import json
import glob
import shutil
import subprocess
import time
import re
import logging
import pandas as pd
import lancedb
from abc import ABC, abstractmethod
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pathlib import Path

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("embeddings_generator")


class RepositoryConfig(BaseModel):
    name: str
    type: Literal["git", "local"] = "git"
    url: Optional[str] = None
    branch: str = "main"
    path: Optional[str] = None
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)
    enabled: bool = True
    last_commit: dict[str, str] = Field(default_factory=dict)


class SourcesConfig(BaseModel):
    repositories: list[RepositoryConfig]


class RepoStats(BaseModel):
    files_indexed: int
    chunks_extracted: int
    vector_calculation_duration_seconds: float
    table_size_bytes: int
    previous_commit: Optional[str] = None
    current_commit: Optional[str] = None
    files_changed_in_diff: list[str] = Field(default_factory=list)


class BuildStats(BaseModel):
    model: str
    mode: str
    start_time: str
    repositories: dict[str, RepoStats]
    total_files_indexed: int
    total_chunks_extracted: int
    total_duration_seconds: float
    total_database_size_bytes: int


import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

class BaseEmbeddingGenerator(ABC):
    def __init__(self, model_name: str, dimensions: int):
        self.model_name = model_name
        self.dimensions = dimensions
        self.config_path = Path(__file__).resolve().parent / "sources.json"
        self.staging_base = Path(".staging-repos")
        self.db_base = Path(f"./embeddings/{self.model_name}")
        
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.api_url = f"{self.ollama_host}/api/embeddings"

        self.session = requests.Session()
        # Retry logic: 5 total attempts, exponential backoff (e.g. 1s, 2s, 4s, 8s, 16s), forcelist status codes
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get_folder_size_bytes(self, folder_path: Path) -> int:
        if not folder_path.exists():
            return 0
        total_size = 0
        for fp in folder_path.rglob("*"):
            if fp.is_file() and not fp.is_symlink():
                total_size += fp.stat().st_size
        return total_size

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

    def get_current_commit(self, repo_path: Path) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting HEAD commit for {repo_path}: {e}")
            return ""

    def sync_repo(self, repo_config: RepositoryConfig) -> Path:
        name = repo_config.name
        url = repo_config.url
        branch = repo_config.branch
        target_dir = self.staging_base / name

        if not url:
            logger.error(f"Skipping repository '{name}': git URL is required for git type sources.")
            return target_dir

        self.staging_base.mkdir(parents=True, exist_ok=True)

        if not target_dir.exists():
            logger.info(f"Cloning {name} from {url} (branch: {branch})...")
            try:
                subprocess.run(
                    ["git", "clone", "--branch", branch, url, str(target_dir)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Error cloning repository {name}: {e}")
        else:
            logger.info(f"Updating cached staging tree for {name}...")
            try:
                subprocess.run(
                    ["git", "fetch", "origin", branch],
                    cwd=str(target_dir),
                    check=True,
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{branch}"],
                    cwd=str(target_dir),
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Error updating repository {name}: {e}. Retrying with fresh clone..."
                )
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                try:
                    subprocess.run(
                        ["git", "clone", "--branch", branch, url, str(target_dir)],
                        check=True,
                    )
                except subprocess.CalledProcessError as clone_err:
                    logger.error(f"Error on fallback clone for {name}: {clone_err}")
        return target_dir

    def get_files_to_index(
        self, base_path: Path, includes: list[str], excludes: list[str]
    ) -> list[Path]:
        matched_files = set()
        for pattern in includes:
            for p in base_path.glob(pattern):
                if p.is_file():
                    matched_files.add(p.resolve())

        for pattern in excludes:
            for p in base_path.glob(pattern):
                p_res = p.resolve()
                if p_res in matched_files:
                    matched_files.remove(p_res)

        return sorted(list(matched_files))

    def run(self, incremental: bool = False):
        start_time = time.time()
        mode_str = "INCREMENTAL" if incremental else "FULL"
        logger.info(
            f"Starting highly modular embedding pipeline ({mode_str} mode) for model tier: {self.model_name}"
        )
        if not self.config_path.exists():
            logger.error(f"Configuration manifest {self.config_path} not located.")
            return

        try:
            config = SourcesConfig.model_validate_json(self.config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse configuration sources: {e}")
            return

        self.db_base.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.db_base))

        repositories_stats = {}
        total_files_indexed = 0
        total_chunks_extracted = 0

        for repo in config.repositories:
            if not repo.enabled:
                logger.info(f"Skipping disabled repository entry: {repo.name}")
                continue

            name = repo.name
            repo_type = repo.type

            if repo_type == "git":
                base_path = self.sync_repo(repo).resolve()
            else:
                base_path = Path(repo.path or ".").resolve()

            if not base_path.exists():
                logger.error(f"Repository path '{base_path}' does not exist.")
                continue

            includes = repo.include
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
            excludes = repo.exclude + global_excludes

            all_files = self.get_files_to_index(base_path, includes, excludes)
            base_table_name = "".join(
                [c if c.isalnum() or c in "-_" else "_" for c in name]
            )

            files_to_process = all_files
            is_fallback_full = False
            current_commit = ""

            if repo_type == "git":
                current_commit = self.get_current_commit(base_path)

            repo_stats = {
                "files_indexed": 0,
                "chunks_extracted": 0,
                "vector_calculation_duration_seconds": 0.0,
                "table_size_bytes": 0,
                "previous_commit": None,
                "current_commit": current_commit if repo_type == "git" else None,
                "files_changed_in_diff": []
            }

            if incremental:
                modified_files = []
                if repo_type == "git":
                    last_commits = repo.last_commit
                    last_commit = last_commits.get(self.model_name)
                    repo_stats["previous_commit"] = last_commit

                    if last_commit:
                        logger.info(
                            f"Calculating diff between {last_commit[:7]} and {current_commit[:7] or 'HEAD'} for {name}..."
                        )
                        try:
                            res = subprocess.run(
                                ["git", "diff", "--name-only", last_commit, "HEAD"],
                                cwd=str(base_path),
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            for line in res.stdout.splitlines():
                                rel_line = line.strip()
                                fp = (base_path / rel_line).resolve()
                                if fp in all_files:
                                    modified_files.append(fp)
                                    repo_stats["files_changed_in_diff"].append(rel_line)
                                elif not fp.exists():
                                    modified_files.append(fp)
                                    repo_stats["files_changed_in_diff"].append(rel_line)
                        except subprocess.CalledProcessError as e:
                            logger.error(
                                f"Git diff failed for {name} (probably missing history): {e}. Falling back to full rebuild."
                            )
                            is_fallback_full = True
                    else:
                        logger.info(
                            f"No previous commit recorded for {name} and model {self.model_name}. Falling back to full rebuild."
                        )
                        is_fallback_full = True

                if not is_fallback_full:
                    if len(modified_files) > 200:
                        logger.warning(
                            f"Over 200 modified files ({len(modified_files)}) detected in repository '{name}'. Escalating to FULL recalculation mode to preserve spatial integrity..."
                        )
                        is_fallback_full = True
                        files_to_process = all_files
                    elif modified_files:
                        logger.info(
                            f"Incremental sync: isolated {len(modified_files)} modified/deleted source files in repository '{name}'."
                        )
                        files_to_process = [
                            f for f in modified_files if f.exists()
                        ]
                        try:
                            table = db.open_table(base_table_name)
                            for mf in modified_files:
                                rel_p = str(mf.relative_to(base_path))
                                table.delete(f"file_path = '{rel_p}'")
                        except Exception as e:
                            logger.error(f"Error opening table or deleting stale records: {e}")
                    else:
                        logger.info(
                            f"Incremental sync: zero modified source files detected in repository '{name}'. Skipping build."
                        )
                        if repo_type == "git" and current_commit:
                            repo.last_commit[self.model_name] = current_commit
                        continue

            logger.info(
                f"Discovered {len(files_to_process)} target sources to process inside index: {name}"
            )

            repo_stats["files_indexed"] = len(files_to_process)
            if not incremental or is_fallback_full:
                repo_stats["files_changed_in_diff"] = ["FULL_REBUILD"]

            repo_chunks = []
            for file_path in files_to_process:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                rel_path = str(file_path.relative_to(base_path))
                if file_path.suffix.lower() == ".html":
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
            repo_stats["chunks_extracted"] = total_chunks
            if total_chunks == 0:
                logger.info(
                    f"No valid extracted text chunks found for repository '{name}'."
                )
                continue

            logger.info(
                f"Extracted {total_chunks} semantic chunks from repository '{name}'. Initiating vectorized bulk embedding phase..."
            )

            records = []
            batch_size = 128
            for b_start in range(0, total_chunks, batch_size):
                b_chunks = repo_chunks[b_start : b_start + batch_size]
                b_texts = [c[0] for c in b_chunks]

                b_start_time = time.time()
                b_vectors = self.generate_vectors(b_texts, is_query=False)

                b_duration = time.time() - b_start_time
                repo_stats["vector_calculation_duration_seconds"] += b_duration
                logger.info(
                    f"Processed vector slice [{b_start + 1} - {b_start + len(b_chunks)} / {total_chunks}] in {b_duration:.2f}s"
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

            if records:
                df_records = pd.DataFrame(records)
                target_mode = (
                    "append" if (incremental and not is_fallback_full) else "overwrite"
                )

                logger.info(
                    f"Committing consolidated table '{base_table_name}' ({target_mode} mode)..."
                )
                try:
                    table = db.create_table(
                        base_table_name, data=df_records, mode=target_mode
                    )
                    table.optimize()
                except Exception:
                    table = db.create_table(
                        base_table_name, data=df_records, mode="overwrite"
                    )

                table_dir = self.db_base / f"{base_table_name}.lance"
                repo_stats["table_size_bytes"] = self._get_folder_size_bytes(table_dir)

            if repo_type == "git" and current_commit:
                repo.last_commit[self.model_name] = current_commit

            repositories_stats[name] = RepoStats(**repo_stats)
            total_files_indexed += repo_stats["files_indexed"]
            total_chunks_extracted += repo_stats["chunks_extracted"]

        self.config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

        build_stats = BuildStats(
            model=self.model_name,
            mode=mode_str,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
            repositories=repositories_stats,
            total_files_indexed=total_files_indexed,
            total_chunks_extracted=total_chunks_extracted,
            total_duration_seconds=round(time.time() - start_time, 2),
            total_database_size_bytes=self._get_folder_size_bytes(self.db_base)
        )

        stats_file = self.db_base / "build_stats.json"
        stats_file.write_text(build_stats.model_dump_json(indent=2), encoding="utf-8")

        print("\n" + "="*80)
        print(f" EMBEDDINGS REBUILD STATISTICS ({self.model_name.upper()} - {mode_str})")
        print("="*80)
        print(f"{'Repository':<25} | {'Files':<8} | {'Chunks':<8} | {'Vec Time (s)':<12} | {'Size':<12}")
        print("-"*80)
        for r_name, r_stats_obj in build_stats.repositories.items():
            size_str = f"{r_stats_obj.table_size_bytes / (1024*1024):.2f} MB"
            print(f"{r_name:<25} | {r_stats_obj.files_indexed:<8} | {r_stats_obj.chunks_extracted:<8} | {r_stats_obj.vector_calculation_duration_seconds:<12.2f} | {size_str:<12}")
        print("-"*80)
        total_size_str = f"{build_stats.total_database_size_bytes / (1024*1024):.2f} MB"
        print(f"{'TOTAL':<25} | {build_stats.total_files_indexed:<8} | {build_stats.total_chunks_extracted:<8} | {build_stats.total_duration_seconds:<12.2f} | {total_size_str:<12}")
        print("="*80 + "\n")

        # Generate and print recommended Git commit message
        print("="*80)
        print(" RECOMMENDED GIT COMMIT MESSAGE FOR DATA UPDATE")
        print("="*80)
        
        msg_parts = []
        msg_parts.append(f"data: sync databases for model {self.model_name} ({mode_str.lower()})")
        msg_parts.append("")
        
        for r_name, r_stats_obj in build_stats.repositories.items():
            prev = r_stats_obj.previous_commit
            curr = r_stats_obj.current_commit
            diff_files = r_stats_obj.files_changed_in_diff
            
            if curr:
                if prev:
                    msg_parts.append(f"{r_name}: synced {prev[:7]} -> {curr[:7]}")
                else:
                    msg_parts.append(f"{r_name}: built fresh from commit {curr[:7]}")
                
                if diff_files:
                    if diff_files == ["FULL_REBUILD"]:
                        msg_parts.append("  Full rebuild triggered.")
                    elif len(diff_files) > 5:
                        msg_parts.append(f"  Files changed ({len(diff_files)} total):")
                        for f in diff_files[:5]:
                            msg_parts.append(f"    - {f}")
                        msg_parts.append("    - ...")
                    else:
                        msg_parts.append("  Files changed:")
                        for f in diff_files:
                            msg_parts.append(f"    - {f}")
            else:
                msg_parts.append(f"{r_name}: synced local files")
            msg_parts.append("")
            
        commit_msg = "\n".join(msg_parts).strip()
        print(commit_msg)
        print("="*80 + "\n")
