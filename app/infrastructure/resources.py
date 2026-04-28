from __future__ import annotations

from functools import lru_cache
from threading import Lock


_CROSS_ENCODER_MODELS: dict[str, object] = {}
_CROSS_ENCODER_LOCK = Lock()


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading Sentence Transformer model: {exc}")
        return None


def get_cross_encoder_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    cached = _CROSS_ENCODER_MODELS.get(model_name)
    if cached is not None:
        return cached

    # Mixed-branch analysis can request the cross-encoder from multiple threads
    # on the first uncached run. Serialize first load and only cache success.
    with _CROSS_ENCODER_LOCK:
        cached = _CROSS_ENCODER_MODELS.get(model_name)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading CrossEncoder model '{model_name}': {exc}")
            return None

        _CROSS_ENCODER_MODELS[model_name] = model
        return model


@lru_cache(maxsize=1)
def get_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading spaCy model: {exc}.")
        return None
