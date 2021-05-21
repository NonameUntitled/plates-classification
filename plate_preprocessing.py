import os
import cv2
import numpy as np

from typing import Tuple


class PlatePreprocessing:
    def __init__(self):
        # TODO add params
        pass

    def preprocess(self, img: np.ndarray, path_to_save: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _subtract_background(self, img: np.ndarray):
        pass

    def _find_plate_circle(self, img: np.ndarray):
        pass

    def _crop_circle_center(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

# TODO add functionality
