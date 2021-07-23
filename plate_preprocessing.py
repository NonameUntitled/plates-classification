import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from constants import *


class ForegroundExtractor:
    def __init__(self):
        # Mask params
        self.FOREGROUND_RADIUS = 50
        self.BACKGROUND_HEIGHT = 60
        self.BACKGROUND_WIDTH = 60
        self.BACKGROUND_DELTA = 15
        self.EMPTINESS_THRESHOLD = 0.01

        self.rotations_list = [
            (None, None),
            (cv2.ROTATE_180, cv2.ROTATE_180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE),
            (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

    def _apply_grab_cut(self, input_image):
        masks = []

        for forward_rot, backward_rot in self.rotations_list:
            # Rotate image
            img_rotated = input_image.copy()
            if forward_rot is not None:
                img_rotated = cv2.rotate(img_rotated, forward_rot)
            img_h, img_w = img_rotated.shape[:2]

            # Define mask for GrabCut
            mask = np.ones((img_h, img_w), dtype=np.uint8) * 3
            background_rectangles = [
                (0, 0, self.BACKGROUND_WIDTH, self.BACKGROUND_HEIGHT),
                (img_w - self.BACKGROUND_WIDTH, 0, img_w, self.BACKGROUND_HEIGHT),
                (0, img_h - self.BACKGROUND_HEIGHT, self.BACKGROUND_WIDTH, img_h),
                (img_w - self.BACKGROUND_WIDTH, img_h - self.BACKGROUND_HEIGHT, img_w, img_h),
                (0, 0, self.BACKGROUND_WIDTH, self.BACKGROUND_DELTA),
                (0, 0, self.BACKGROUND_DELTA, self.BACKGROUND_HEIGHT),
                (0, img_h - self.BACKGROUND_DELTA, img_w, img_h),
                (img_w - self.BACKGROUND_DELTA, 0, img_w, img_h)
            ]
            for bg_rect in background_rectangles:
                cv2.rectangle(mask, bg_rect[:2], bg_rect[2:], cv2.GC_BGD, -1)
            cv2.circle(mask, (int(round(img_w / 2)), int(round(img_h / 2))), self.FOREGROUND_RADIUS, cv2.GC_FGD, -1)

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Apply grab cut
            new_mask, new_bgd_model, new_fgd_model = cv2.grabCut(
                img_rotated, mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK  # TODO find best iter cnt
            )

            # Create mask for initial image
            fixed_mask = np.where((new_mask == 2) | (new_mask == 0), 0, 255).astype('uint8')
            if backward_rot is not None:
                fixed_mask = cv2.rotate(fixed_mask, backward_rot)

            masks.append(fixed_mask)

        # Create final mask for image
        masks = np.where(np.array(masks, dtype=np.uint8) == 255, 1, 0).astype('uint8')
        combined_mask = np.cumprod(masks, axis=0)[-1].astype(np.uint8)
        return combined_mask

    @staticmethod
    def _select_main_area_only(grab_cut_mask):
        # Find the biggest contour
        mask_new = grab_cut_mask * 255
        contours, _ = cv2.findContours(mask_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)

        # Remove other contours
        for y in range(grab_cut_mask.shape[0]):
            for x in range(grab_cut_mask.shape[1]):
                result = cv2.pointPolygonTest(biggest_contour, (x, y), False)
                if result >= 0:
                    grab_cut_mask[y, x] = 255
                else:
                    grab_cut_mask[y, x] = 0
        return grab_cut_mask, biggest_contour

    def __call__(self, input_image):
        img_orig = input_image.copy()

        # Apply Grab Cut
        grab_cut_mask = self._apply_grab_cut(input_image)

        # Select only plate area
        grab_cut_mask, biggest_contour = self._select_main_area_only(grab_cut_mask)
        masked_img = cv2.bitwise_and(img_orig, img_orig, mask=grab_cut_mask)

        return masked_img


def train_val_merge(path_to_train: str, path_to_val: str):
    for dir_name in os.listdir(path_to_train):
        train_dir_path = os.path.join(path_to_train, dir_name)
        val_dir_path = os.path.join(path_to_val, dir_name)

        for filename in os.listdir(val_dir_path):
            val_filepath = os.path.join(val_dir_path, filename)
            train_filepath = os.path.join(train_dir_path, filename)
            shutil.move(val_filepath, train_filepath)


def train_val_split(path_to_train: str, path_to_val: str, val_size: float = 0.3):
    for dir_name in os.listdir(path_to_train):
        file_counter = 0

        train_dir_path = os.path.join(path_to_train, dir_name)
        val_dir_path = os.path.join(path_to_val, dir_name)

        files = os.listdir(train_dir_path)
        np.random.shuffle(files)

        for filename in files:
            file_counter += val_size
            if file_counter >= 1:
                train_filepath = os.path.join(train_dir_path, filename)
                val_filepath = os.path.join(val_dir_path, filename)
                shutil.move(train_filepath, val_filepath)
                file_counter -= 1


if __name__ == '__main__':
    Path(CLEANED_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
    Path(DIRTY_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_TEST_PATH).mkdir(parents=True, exist_ok=True)

    train_val_merge(path_to_train=PROCESSED_TRAIN_PATH, path_to_val=PROCESSED_VAL_PATH)

    process_types = ['cleaned', 'dirty', 'test']  # 'cleaned', 'dirty', 'test'

    foreground_extractor = ForegroundExtractor()

    for process_type in process_types:
        if process_type == 'cleaned':
            data_path = CLEANED_PATH
            save_path = CLEANED_PROCESSED_PATH
        elif process_type == 'dirty':
            data_path = DIRTY_PATH
            save_path = DIRTY_PROCESSED_PATH
        else:
            data_path = TEST_PATH
            save_path = PROCESSED_TEST_PATH

        for idx, filename in enumerate(tqdm(os.listdir(data_path))):
            filepath = os.path.join(data_path, filename)
            img = cv2.imread(filepath)
            img_center = foreground_extractor(img)

            new_filepath = os.path.join(save_path, f'{process_type}_{filename.split(".")[0]}.png')
            cv2.imwrite(new_filepath, img_center)

    train_val_split(path_to_train=PROCESSED_TRAIN_PATH, path_to_val=PROCESSED_VAL_PATH)
