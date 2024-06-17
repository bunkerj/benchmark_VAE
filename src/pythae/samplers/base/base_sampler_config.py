from pydantic.dataclasses import dataclass

from src.pythae.config import BaseConfig


@dataclass
class BaseSamplerConfig(BaseConfig):
    """
    BaseSampler config class.
    """

    pass
