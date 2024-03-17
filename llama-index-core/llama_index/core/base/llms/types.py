from enum import Enum
from typing import Any, AsyncGenerator, Generator, Optional, Union, List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


# ===== Generic Model Input - Chat =====
class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[Any] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> "ChatMessage":
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, content=content, **kwargs)


class LogProb(BaseModel):
    """LogProb of a token."""

    token: str = Field(default_factory=str)
    logprob: float = Field(default_factory=float)
    bytes: List[int] = Field(default_factory=list)


class StopReason(str, Enum):
    """Reason for stopping the generation."""

    MAX_TOKENS = "max_tokens"
    LENGTH = "length"
    STOP_SEQUENCE = "stop_sequence"
    END_OF_TEXT = "end_of_text"


# ===== Common metadata =====


class ResponseMeta(BaseModel):
    """Response metadata."""

    input_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in the input prompt.",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens in the output response.",
    )
    stop_reason: Optional[StopReason] = Field(
        default=None,
        description="Reason for stopping the generation.",
    )
    stop_sequence: Optional[str] = Field(
        default=None,
        description="Sequence that caused the generation to stop.",
    )
    delta: Optional[str] = Field(
        default=None,
        description="New text that just streamed in (only relevant when streaming).",
    )
    logprobs: Optional[List[List[LogProb]]] = Field(
        default=None,
        description="Log probabilities of tokens in the response.",
    )


# ===== Generic Model Output - Chat =====
class ChatResponse(BaseModel):
    """Chat response."""

    message: ChatMessage
    raw: Optional[dict] = None
    delta: Optional[str] = None
    logprobs: Optional[List[List[LogProb]]] = None
    additional_kwargs: dict = Field(default_factory=dict)

    metadata: ResponseMeta = Field(default_factory=ResponseMeta)

    def __str__(self) -> str:
        return str(self.message)


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]


# ===== Generic Model Output - Completion =====
class CompletionResponse(BaseModel):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
        metadata: Response metadata.
    """

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[dict] = None
    logprobs: Optional[List[List[LogProb]]] = None
    delta: Optional[str] = None

    metadata: ResponseMeta = Field(default_factory=ResponseMeta)

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]


class LLMMetadata(BaseModel):
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input and output for one response."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )
    system_role: MessageRole = Field(
        default=MessageRole.SYSTEM,
        description="The role this specific LLM provider"
        "expects for system prompt. E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere",
    )
