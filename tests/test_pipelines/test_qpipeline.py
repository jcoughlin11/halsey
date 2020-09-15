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
