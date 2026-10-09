"""Tests for infra.llm.client — OpenAIMediaClient and MediaFactory.

Covers:
- MediaFactory.create() with different providers (openai/seedream/tokenfree/unknown)
- OpenAIMediaClient image generation (async + sync)
- OpenAIMediaClient video generation + polling
- OpenAIMediaClient audio synthesis
- Provider name preservation for seedream
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from infra.llm.client import (
    MediaFactory,
    OpenAIMediaClient,
    get_media_client,
    init_media_client,
    _reset_media_client_for_test,
)
from infra.llm.models import MediaRequest, MediaResponse, ImageResult, VideoResult, AudioResult


class TestMediaFactory:
    """Tests for MediaFactory.create() provider selection."""

    def test_create_openai_provider(self):
        client = MediaFactory.create({
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "image_model": "dall-e-3",
            "video_model": "sora",
            "audio_model": "tts-1",
        })
        assert isinstance(client, OpenAIMediaClient)
        assert client.provider == "openai"
        assert client.image_model == "dall-e-3"
        assert client.video_model == "sora"
        assert client.audio_model == "tts-1"

    def test_create_seedream_provider(self):
        """Seedream is OpenAI-compatible — uses OpenAIMediaClient with seedream provider name."""
        client = MediaFactory.create({
            "provider": "seedream",
            "api_key": "test-seedream-key",
            "base_url": "https://seedream-api.com/v1",
            "image_model": "seedream-v2",
            "video_model": "seedream-video",
        })
        assert isinstance(client, OpenAIMediaClient)
        assert client.api_key == "test-seedream-key"
        assert client.base_url == "https://seedream-api.com/v1"
        assert client.image_model == "seedream-v2"
        assert client.video_model == "seedream-video"
        assert client.provider == "seedream"

    def test_create_tokenfree_provider(self):
        """Tokenfree is also OpenAI-compatible."""
        client = MediaFactory.create({
            "provider": "tokenfree",
            "api_key": "tokenfree-key",
        })
        assert isinstance(client, OpenAIMediaClient)
        assert client.api_key == "tokenfree-key"

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unsupported media provider"):
            MediaFactory.create({
                "provider": "unknown",
                "api_key": "test-key",
            })

    def test_create_without_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            MediaFactory.create({
                "provider": "openai",
            })

    def test_create_case_insensitive_provider(self):
        """MediaFactory lowercases the provider name."""
        client = MediaFactory.create({
            "provider": "OPenAI",
            "api_key": "test-key",
        })
        assert client.provider == "openai"


class TestOpenAIMediaClientImage:
    """Tests for OpenAIMediaClient generate_image methods."""

    @pytest.fixture
    def media_client(self):
        client = OpenAIMediaClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            image_model="dall-e-3",
        )
        # Replace the async client with a mock for deterministic testing
        client._async_client = MagicMock()
        client._async_client.images = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_generate_image_async_returns_images(self, media_client):
        """generate_image returns ImageResult list from API response."""
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.url = "https://example.com/image.png"
        mock_item.b64_json = None
        mock_item.revised_prompt = "revised prompt"
        mock_response.data = [mock_item]
        media_client._async_client.images.generate = AsyncMock(return_value=mock_response)

        result = await media_client.generate_image(
            MediaRequest(prompt="a cat", model="dall-e-3")
        )

        assert result.modality == "image"
        assert result.model == "dall-e-3"
        assert len(result.images) == 1
        assert result.images[0].url == "https://example.com/image.png"
        assert result.images[0].revised_prompt == "revised prompt"

    @pytest.fixture
    def sync_media_client(self):
        client = OpenAIMediaClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            image_model="dall-e-3",
        )
        client._sync_client = MagicMock()
        client._sync_client.images = MagicMock()
        return client

    def test_generate_image_sync_returns_images(self, sync_media_client):
        """generate_image_sync returns ImageResult list from API response."""
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.url = "https://example.com/image.png"
        mock_item.b64_json = None
        mock_item.revised_prompt = None
        mock_response.data = [mock_item]
        sync_media_client._sync_client.images.generate = lambda **kwargs: mock_response

        result = sync_media_client.generate_image_sync(
            MediaRequest(prompt="a dog", model="dall-e-3")
        )

        assert result.modality == "image"
        assert len(result.images) == 1
        assert result.images[0].url == "https://example.com/image.png"

    @pytest.mark.asyncio
    async def test_generate_image_passes_size(self, media_client):
        """generate_image passes size parameter when provided."""
        mock_response = MagicMock()
        mock_response.data = []
        media_client._async_client.images.generate = AsyncMock(return_value=mock_response)

        await media_client.generate_image(MediaRequest(prompt="test", size="1024x1024", n=2))

        _, kwargs = media_client._async_client.images.generate.call_args
        assert kwargs["size"] == "1024x1024"
        assert kwargs["n"] == 2

    @pytest.mark.asyncio
    async def test_generate_image_passes_reference_images(self, media_client):
        """generate_image passes reference image URLs as extra_image_urls."""
        mock_response = MagicMock()
        mock_response.data = []
        media_client._async_client.images.generate = AsyncMock(return_value=mock_response)

        await media_client.generate_image(
            MediaRequest(prompt="test", reference_images=["https://img.com/ref.png"])
        )

        _, kwargs = media_client._async_client.images.generate.call_args
        assert kwargs["extra_image_urls"] == ["https://img.com/ref.png"]

    @pytest.mark.asyncio
    async def test_generate_image_uses_retry_config(self, media_client):
        """generate_image uses retry configuration."""
        mock_response = MagicMock()
        mock_response.data = []
        media_client._async_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("infra.llm.client.retry_async", new=AsyncMock(return_value=mock_response)) as mock_retry:
            await media_client.generate_image(MediaRequest(prompt="test"))

        assert mock_retry.called


class TestOpenAIMediaClientVideo:
    """Tests for OpenAIMediaClient generate_video + poll_video."""

    @pytest.fixture
    def video_client(self):
        return OpenAIMediaClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            image_model="dall-e-3",
            video_model="sora",
        )

    @pytest.mark.asyncio
    async def test_generate_video_returns_task_id(self, video_client):
        """generate_video submits task and returns task_id in pending status."""
        mock_response = MagicMock()
        mock_response.id = "video-task-123"

        mock_video = MagicMock()
        mock_video.generate = AsyncMock(return_value=mock_response)
        with patch("infra.llm.client.retry_async", new=AsyncMock(return_value=mock_response)), \
             patch.object(video_client._async_client, "video", mock_video, create=True):
            result = await video_client.generate_video(
                MediaRequest(prompt="a beautiful landscape", model="sora")
            )

        assert result.modality == "video"
        assert result.videos[0].task_id == "video-task-123"
        assert result.videos[0].status == "pending"

    @pytest.mark.asyncio
    async def test_generate_video_fallback_to_chat(self, video_client):
        """generate_video falls back to chat if video.generate not available."""
        mock_response = MagicMock()
        mock_response.id = "chat-task-456"

        mock_video = MagicMock()
        mock_video.generate = AsyncMock(side_effect=AttributeError("no video"))

        async def mock_retry_fn(fn, config=None):
            return await fn()

        with patch("infra.llm.client.retry_async", new=AsyncMock(side_effect=mock_retry_fn)), \
             patch.object(video_client._async_client, "video", mock_video, create=True), \
             patch.object(video_client._async_client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await video_client.generate_video(
                MediaRequest(prompt="test video", model="sora")
            )

        assert result.videos[0].task_id == "chat-task-456"

    @pytest.mark.asyncio
    async def test_generate_video_without_model_raises(self):
        """generate_video raises ValueError if video_model not configured."""
        client = OpenAIMediaClient(api_key="test", image_model="dall-e-3")
        with pytest.raises(ValueError, match="video_model not configured"):
            await client.generate_video(MediaRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_poll_video_completed(self, video_client):
        """poll_video returns video_url when status is completed."""
        mock_response = MagicMock()
        mock_response.status = "completed"
        mock_response.video_url = "https://example.com/video.mp4"

        mock_tasks = MagicMock()
        mock_tasks.retrieve = AsyncMock(return_value=mock_response)
        mock_video = MagicMock()
        mock_video.tasks = mock_tasks
        with patch.object(video_client._async_client, "video", mock_video, create=True):
            result = await video_client.poll_video("video-task-123")

        assert result.modality == "video"
        assert result.videos[0].status == "completed"
        assert result.videos[0].video_url == "https://example.com/video.mp4"

    @pytest.mark.asyncio
    async def test_poll_video_failed(self, video_client):
        """poll_video returns error when status is failed."""
        mock_response = MagicMock()
        mock_response.status = "failed"
        mock_response.error = "Generation failed"
        mock_response.completed = False
        mock_response.failed = False
        mock_response.video_url = None
        mock_response.result_url = None

        mock_tasks = MagicMock()
        mock_tasks.retrieve = AsyncMock(return_value=mock_response)
        mock_video = MagicMock()
        mock_video.tasks = mock_tasks
        with patch.object(video_client._async_client, "video", mock_video, create=True):
            result = await video_client.poll_video("video-task-123")

        assert result.videos[0].status == "failed"
        assert result.videos[0].error == "Generation failed"

    @pytest.mark.asyncio
    async def test_poll_video_timeout(self, video_client):
        """poll_video returns running status after timeout."""
        mock_response = MagicMock()
        mock_response.status = "running"
        mock_response.completed = False
        mock_response.failed = False
        mock_response.video_url = None
        mock_response.result_url = None

        mock_tasks = MagicMock()
        mock_tasks.retrieve = AsyncMock(return_value=mock_response)
        mock_video = MagicMock()
        mock_video.tasks = mock_tasks
        with patch.object(video_client._async_client, "video", mock_video, create=True):
            result = await video_client.poll_video("video-task-123", interval=0.01, max_wait=0.05)

        assert result.videos[0].status == "running"
        assert "timed out" in (result.videos[0].error or "").lower()


class TestOpenAIMediaClientAudio:
    """Tests for OpenAIMediaClient synthesize_audio."""

    @pytest.fixture
    def audio_client(self):
        return OpenAIMediaClient(
            api_key="test-key",
            audio_model="tts-1",
        )

    @pytest.mark.asyncio
    async def test_synthesize_audio_from_url(self, audio_client):
        """synthesize_audio returns url when provider returns URL."""
        mock_response = MagicMock()
        mock_response.url = "https://example.com/audio.mp3"
        mock_response.read = None

        with patch("infra.llm.client.retry_async", new=AsyncMock(return_value=mock_response)):
            with patch.object(audio_client._async_client.audio.speech, "create", new=AsyncMock(return_value=mock_response)):
                result = await audio_client.synthesize_audio(
                    MediaRequest(prompt="Hello world", model="tts-1")
                )

        assert result.modality == "audio"
        assert result.audios[0].url == "https://example.com/audio.mp3"

    @pytest.mark.asyncio
    async def test_synthesize_audio_from_bytes(self, audio_client):
        """synthesize_audio encodes binary response to b64."""
        mock_response = MagicMock()
        mock_response.url = None
        mock_response.read = lambda: b"fake-audio-bytes"

        with patch("infra.llm.client.retry_async", new=AsyncMock(return_value=mock_response)):
            with patch.object(audio_client._async_client.audio.speech, "create", new=AsyncMock(return_value=mock_response)):
                result = await audio_client.synthesize_audio(
                    MediaRequest(prompt="Hello world", model="tts-1")
                )

        assert result.audios[0].b64_json is not None
        assert result.audios[0].url is None

    @pytest.mark.asyncio
    async def test_synthesize_audio_without_model_raises(self):
        """synthesize_audio raises ValueError if audio_model not configured."""
        client = OpenAIMediaClient(api_key="test", image_model="dall-e-3")
        with pytest.raises(ValueError, match="audio_model not configured"):
            await client.synthesize_audio(MediaRequest(prompt="test"))


class TestMediaClientGlobal:
    """Tests for init_media_client / get_media_client globals."""

    def test_init_media_client_returns_none_without_key(self):
        """init_media_client returns None when api_key not configured."""
        _reset_media_client_for_test()
        result = asyncio.new_event_loop().run_until_complete(
            init_media_client({"media": {}})
        )
        assert result is None
        assert get_media_client() is None

    @pytest.mark.asyncio
    async def test_init_media_client_with_config(self):
        """init_media_client creates client when api_key provided."""
        _reset_media_client_for_test()
        result = await init_media_client({
            "media": {
                "provider": "seedream",
                "api_key": "seedream-test-key",
                "base_url": "https://seedream.com/v1",
                "image_model": "seedream-v2",
            }
        })
        assert result is not None
        assert result.api_key == "seedream-test-key"
        _reset_media_client_for_test()

    def test_get_media_client_returns_none_before_init(self):
        """get_media_client returns None before initialization."""
        _reset_media_client_for_test()
        assert get_media_client() is None
