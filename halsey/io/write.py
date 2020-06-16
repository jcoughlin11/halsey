"""
Title: write.py
Notes:
"""
import numpy as np

from halsey.utils.misc import lock_file


# ============================================
#                save_model
# ============================================
def save_model(manager, trainer):
    """
    Doc string.

    Structure of output directory:
    |-output/
        |-param.lock
        |-checkpoint_1/
        |-...
        |-checkpoint_N/
            |-tf checkpoint manager file(s)
            |-object-specific parameter file(s)
            |-game env
            |-replay buffer file(s)

    Saving multiple copies of the full replay buffer can get expensive, so
    maybe just save the diff? That is, save the full buffer for the first
    checkpoint and then, for each subsequent checkpoint, just save the new
    experiences
    """
    lock_file(manager.clArgs.paramFile)
    save_emulator(trainer.game.env)
    save_training_state()
    trainer.checkpointManager.save()
    save_replay_buffer()


# ============================================
#               save_emulator
# ============================================
def save_emulator(env):
    """
    Doc string.

    NOTE: This really only works with deterministic environments.
    """
    envState = env.unwrapped.clone_full_state()
    np.save("envState", envState)


# ============================================
#             save_training_state
# ============================================
def save_training_state():
    """
    Doc string.

    This is the counters and other misc params (current episode, etc.)
    """
    raise NotImplementedError


# ============================================
#            save_replay_buffer
# ============================================
def save_replay_buffer():
    """
    Doc string.
    """
    raise NotImplementedError
