import gym
import pytest


@pytest.fixture(scope="module")
def frame(request):
    env = gym.make("SpaceInvaders-v4")
    frame = env.reset()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (210, 160, 3)
    return frame


@pytest.fixture(scope="module")
def qPipeline(request):
    params = {
        "traceLen" : 4,
        "offsetHeight" : 8,
        "offsetWidth" : 4,
        "cropHeight" : 110,
        "cropWidth" : 84,
    }
    pipeline = QPipeline(params)
    return pipeline
