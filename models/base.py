"""模型抽象基类。所有模型实现需继承此类并实现 generate 方法。"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all evaluation models."""

    @abstractmethod
    def generate(self, instruction: str, video_path: str, **kwargs) -> str:
        """
        Generate a text response given an instruction and a video.

        Args:
            instruction: The text prompt / question.
            video_path: Absolute path to the video file.

        Returns:
            Generated text response.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return model identifier string (used in output file names)."""
        raise NotImplementedError
