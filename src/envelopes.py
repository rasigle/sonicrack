import numpy as np


def envelope(t, released, release_start):
    """Compute ADSR amplitude at time t."""

    # 🎚️ ADSR envelope settings
    ATTACK = 0.05
    DECAY = 0.1
    SUSTAIN_LEVEL = 0.4
    RELEASE = 0.3

    if not released:
        if t < ATTACK:
            return t / ATTACK
        elif t < ATTACK + DECAY:
            return 1 - (1 - SUSTAIN_LEVEL) * ((t - ATTACK) / DECAY)
        else:
            return SUSTAIN_LEVEL
    else:
        rel_t = t - release_start
        if rel_t < RELEASE:
            return SUSTAIN_LEVEL * (1 - rel_t / RELEASE)
        else:
            return 0.0


def generate_adsr_envelope(t, attack, decay, sustain, release, total_time):
    env = np.zeros_like(t)
    attack_end = attack
    decay_end = attack + decay
    release_start = total_time - release
    for i, ti in enumerate(t):
        if ti < attack_end:
            env[i] = ti / attack_end if attack_end > 0 else 1
        elif ti < decay_end:
            env[i] = 1 - (ti - attack_end) / max(decay, 1e-9) * (1 - sustain)
        elif ti < release_start:
            env[i] = sustain
        else:
            env[i] = sustain * (1 - (ti - release_start) / max(release, 1e-9))
    return env
