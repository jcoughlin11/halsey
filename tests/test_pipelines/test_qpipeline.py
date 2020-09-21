from queue import deque

import numpy as np
import pytest


# ============================================
#            test_normalize_frame
# ============================================
def test_normalize_frame(frame, qPipeline):
    normFrame = qPipeline.normalize_frame(frame)
    assert (normFrame >= 0.0).all() and (normFrame <= 1.0).all()
    assert normFrame.shape == frame.shape


# ============================================
#               test_grayscale
# ============================================
def test_grayscale(frame, qPipeline):
    grayFrame = qPipeline.grayscale(frame)
    assert len(grayFrame.shape) == 3
    assert grayFrame.shape[-1] == 1
    assert grayFrame.shape[:-1] == frame.shape[:-1]


# ============================================
#                 test_crop
# ============================================
def test_crop(frame, qPipeline):
    croppedFrame = qPipeline.crop(frame)
    assert len(croppedFrame.shape) == 3
    assert croppedFrame.shape[0] == qPipeline.params["cropHeight"]
    assert croppedFrame.shape[1] == qPipeline.params[ cropWidth ]
    assert croppedFrame.shape[2] == frame.shape[2]


# ============================================
#                 test_stack
# ============================================
@pytest.mark.parametrize("newEpisode", [True, False])
@pytest.mark.parametrize("dataFormat", ["channels_first", "channels_last"])
def test_stack(frame, qPipeline, newEpisode, dataFormat):
    qPipeline.dataFormat = dataFormat
    # Stack calls tf.squeeze, which means we need to grayscale the
    # frame to ensure the channels dimension is 1
    grayFrame = qPipeline.grayscale(frame)
    # Note that we're calling stack here without having first called
    # either crop or normalize
    stackedFrame = qPipeline.stack(grayFrame, newEpisode)
    assert isinstance(stackedFrame, np.ndarry)
    assert len(stackedFrame.shape) == 3
    if newEpisode:
        if dataFormat == "channels_first":
            assert stackedFrame.shape[0] == qPipeline.params["traceLen"]
            assert stackedFrame.shape[1] == 210
            assert stackedFrame.shape[2] == 160
            assert np.all(stackedFrame == stackedFrame[0,:], axis=0)
        else:
            assert stackedFrame.shape[0] == 210
            assert stackedFrame.shape[1] == 160
            assert stackedFrame.shape[2] == qPipeline.params["traceLen"]
    else:
        if dataFormat == "channels_first":
            assert stackedFrame.shape[0] == 1 
            assert stackedFrame.shape[1] == 210
            assert stackedFrame.shape[2] == 160
        else:
            assert stackedFrame.shape[0] == 210
            assert stackedFrame.shape[1] == 160
            assert stackedFrame.shape[2] == 1 


# ============================================
#                 test_process
# ============================================
@pytest.mark.parametrize("newEpisode", [True, False])
def test_process(frame, qPipeline, newEpisode):
    qPipeline.dataFormat = "channels_first"
    state = qPipeline.process(frame, newEpisode)
    assert isinstance(state, np.ndarray)
    if newEpisode:
        assert state.shape == (4, 110, 84)
    else:
        assert state.shape == (1, 110, 84)
    qPipeline.dataFormat = "channels_last"
    state = qPipeline.process(frame, newEpisode)
    if newEpisode:
        assert state.shape == (110, 84, 4)
    else:
        assert state.shape == (110, 84, 1)
