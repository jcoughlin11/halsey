import gym
import pytest


@pytest.fixture(scope="module")
def frame(request):
    env = gym.make("SpaceInvaders-v4")
    try:
        frame = env.reset()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (210, 160, 3)
        yield frame
    finally:
        env.close()


@pytest.fixture(scope="function")
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
