import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .base_encoder import BaseEmbeddingGenerator

logger = logging.getLogger("embeddings_generator.gemma")


class GemmaGenerator(BaseEmbeddingGenerator):
    def __init__(self):
        super().__init__(model_name="gemma", dimensions=768)

    def _get_splitter(self, file_path: str):
        ext = file_path.split(".")[-1].lower()
        mapping = {
            "py": Language.PYTHON,
            "js": Language.JS,
            "ts": Language.TS,
            "md": Language.MARKDOWN,
            "html": Language.HTML,
            "cpp": Language.CPP,
            "go": Language.GO,
            "rs": Language.RUST,
        }

        chunk_size = 1500
        chunk_overlap = 150

        if ext in mapping:
            try:
                return RecursiveCharacterTextSplitter.from_language(
                    language=mapping[ext],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception:
                pass

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk_text(self, content: str, file_path: str) -> list[str]:
        try:
            splitter = self._get_splitter(file_path)
            return splitter.split_text(content)
        except Exception:
            words = content.split()
            return [
                " ".join(words[i : i + 500])
                for i in range(0, len(words), 500)
                if " ".join(words[i : i + 500]).strip()
            ]

    def generate_vector(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            formatted_text = f"task: search result | query: {text}"
        else:
            formatted_text = f"title: none | text: {text}"

        try:
            resp = self.session.post(
                self.api_url,
                json={"model": "embeddinggemma", "prompt": formatted_text},
                timeout=120,
            )
            resp.raise_for_status()
            res = resp.json()
            if "embedding" in res:
                if not hasattr(self, "_logged_success"):
                    logger.info(
                        f"Successfully generated authentic Gemma vector arrays (dim: {len(res['embedding'])}) via Ollama binding."
                    )
                    self._logged_success = True
                return res["embedding"]
            else:
                raise RuntimeError(
                    f"Ollama API payload error missing embedding structures: {res}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate vector via local Ollama service: {e}"
            ) from e

    def generate_vectors(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(
                executor.map(
                    lambda t: self.generate_vector(t, is_query=is_query), texts
                )
            )
