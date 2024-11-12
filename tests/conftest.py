import pytest
from omegaconf import OmegaConf


@pytest.fixture
def config():
    return OmegaConf.load("configs/train.yaml")