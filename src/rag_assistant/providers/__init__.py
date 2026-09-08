"""Model provider abstraction and implementations."""

from rag_assistant.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    GenerationRequest,
    GenerationResponse,
    PromptSegment,
)
from rag_assistant.providers.extractive import ExtractiveChatProvider
from rag_assistant.providers.hashing import HashingEmbeddingProvider
from rag_assistant.providers.registry import (
    FallbackChatProvider,
    build_chat_provider,
    build_embedding_provider,
    close_all,
)

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "ExtractiveChatProvider",
    "FallbackChatProvider",
    "GenerationRequest",
    "GenerationResponse",
    "HashingEmbeddingProvider",
    "PromptSegment",
    "build_chat_provider",
    "build_embedding_provider",
    "close_all",
]
