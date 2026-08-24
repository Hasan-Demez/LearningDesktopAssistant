import numpy as np
import sounddevice as sd


SAMPLE_RATE = 24000


def play_beep(
    frequency=1000,
    duration=0.25,
    volume=0.4,
    sample_rate=SAMPLE_RATE,
):
    """Play a short beep to signal readiness."""
    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False,
    )

    beep = (
        volume
        * np.sin(2 * np.pi * frequency * t)
    ).astype(np.float32)

    sd.play(beep, sample_rate)
    sd.wait()