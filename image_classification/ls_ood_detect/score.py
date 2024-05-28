import numpy as np
from .detectors import DetectorKDE


def get_hz_scores(hz_detector: DetectorKDE, samples: np.ndarray):
    scores = hz_detector.get_density_scores(samples)
    return scores
