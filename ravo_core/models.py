"""Model identifiers used by the typed and spoken assistants."""

from typing import Final

TYPED_CHAT_MODEL: Final[str] = "gpt-4.1-mini"
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
VOICE_LLM_MODEL: Final[str] = "gpt-4.1-mini"
LEGACY_STT_MODEL: Final[str] = "deepgram/flux-general-en"
BILINGUAL_STT_MODEL: Final[str] = "openai/gpt-live-transcribe"
BILINGUAL_STT_BACKUP_MODEL: Final[str] = "xai/stt-1"
TTS_MODEL: Final[str] = "cartesia/sonic-3"
