import os
import cv2
import numpy as np
from tqdm import tqdm
import pathlib

FOREGROUND_RADIUS = 60
BACKGROUND_HEIGHT = 60
BACKGROUND_WIDTH = 60
BACKGROUND_DELTA = 5

EMPTINESS_THRESHOLD = 0.01


def apply_grab_cut(img):
    masks = []
    rotations_list = [
        (None, None),
        (cv2.ROTATE_180, cv2.ROTATE_180),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE),
        (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for forward_rot, backward_rot in rotations_list:
        img_rotated = img.copy()
        if forward_rot is not None:
            img_rotated = cv2.rotate(img_rotated, forward_rot)

        img_h, img_w = img_rotated.shape[:2]
        mask = np.ones((img_h, img_w), dtype=np.uint8) * 3

        background_rectangles = [
            (0, 0, BACKGROUND_WIDTH, BACKGROUND_HEIGHT),
            (img_w - BACKGROUND_WIDTH, 0, img_w, BACKGROUND_HEIGHT),
            (0, img_h - BACKGROUND_HEIGHT, BACKGROUND_WIDTH, img_h),
            (img_w - BACKGROUND_WIDTH, img_h - BACKGROUND_HEIGHT, img_w, img_h),
            (0, 0, BACKGROUND_WIDTH, BACKGROUND_DELTA),
            (0, 0, BACKGROUND_DELTA, BACKGROUND_HEIGHT),
            (0, img_h - BACKGROUND_DELTA, img_w, img_h),
            (img_w - BACKGROUND_DELTA, 0, img_w, img_h)
        ]

        cv2.circle(mask, (int(round(img_w / 2)), int(round(img_h / 2))), FOREGROUND_RADIUS, cv2.GC_FGD, -1)
        for bg_rect in background_rectangles:
            cv2.rectangle(mask, bg_rect[:2], bg_rect[2:], cv2.GC_BGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        new_mask, new_bgd_model, new_fgd_model = cv2.grabCut(
            img_rotated, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK  # TODO iter cnt
        )

        fixed_mask = np.where((new_mask == 2) | (new_mask == 0), 0, 255).astype('uint8')

        if backward_rot is not None:
            fixed_mask = cv2.rotate(fixed_mask, backward_rot)
        masks.append(fixed_mask)

    combined_mask = np.cumprod(masks, axis=0)[-1].astype(np.uint8)
    return combined_mask


def select_main_area_only(grab_cut_mask):
    mask_new = np.where((grab_cut_mask == 1) + (grab_cut_mask == 3), 255, 0).astype('uint8')
    contours, _ = cv2.findContours(mask_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    for y in range(grab_cut_mask.shape[0]):
        for x in range(grab_cut_mask.shape[1]):
            result = cv2.pointPolygonTest(biggest_contour, (x, y), False)
            if result >= 0:
                grab_cut_mask[y, x] = 255
            else:
                grab_cut_mask[y, x] = 0
    return grab_cut_mask, biggest_contour


def select_best_circle(masked_img, mask):
    img_h, img_w = masked_img.shape[:2]

    moments = cv2.moments(mask)
    mean_x = int(round(moments['m10'] / moments['m00']))
    mean_y = int(round(moments['m01'] / moments['m00']))

    masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        masked_img_gray, cv2.HOUGH_GRADIENT, 1, 3,
        param1=5, param2=5,
        minRadius=int(r00 // 2), maxRadius=int(r00)
    )

    def emptiness_percentage(circle_info):
        center_x, center_y, radius = circle_info
        start_x = max(0, int(center_x - radius / np.sqrt(2)))
        finish_x = min(img_w, int(center_x + radius / np.sqrt(2)))
        start_y = max(0, int(center_y - radius / np.sqrt(2)))
        finish_y = min(img_h, int(center_y + radius / np.sqrt(2)))

        empty_cnt_threshold = int((finish_x - start_x) * (finish_y - start_y) * EMPTINESS_THRESHOLD)
        empty_cnt = 0
        for img_x in range(start_x, finish_x):
            for img_y in range(start_y, finish_y):
                if mask[img_y, img_x] == 0:
                    empty_cnt += 1
                    if empty_cnt >= empty_cnt_threshold:
                        break

        emptiness_ratio = empty_cnt / ((finish_x - start_x) * (finish_y - start_y))
        return emptiness_ratio

    if circles is not None:
        filtered_circles = list(
            filter(
                lambda c:
                abs(c[0] - mean_x) <= 0.1 * img_w and abs(c[1] - mean_y) <= 0.1 * img_h \
                or abs(c[0] - img_w / 2) <= 0.1 * img_w and abs(c[1] - img_h / 2) <= 0.1 * img_h,
                circles[0]
            )
        )
        circles_emptiness_ratio = list(map(emptiness_percentage, filtered_circles))
        circle_info = list(zip(filtered_circles, circles_emptiness_ratio))
        filtered_circle_info = list(filter(lambda circle_info: circle_info[1] < EMPTINESS_THRESHOLD, circle_info))
        best_circle_info = max(filtered_circle_info, key=lambda c: c[0][2])
        best_x, best_y, best_r = best_circle_info[0]
    else:
        best_x, best_y, best_r = img_w // 2, img_h // 2, FOREGROUND_RADIUS

    return best_x, best_y, best_r


if __name__ == '__main__':
    positive_path_ = r'./data/train/cleaned'
    positive_path_save_ = r'./data/train/cleaned_saved'
    negative_path_ = r'./data/train/dirty'
    negative_path_save_ = r'./data/train/dirty_saved'

    pathlib.Path(positive_path_save_).mkdir(parents=True, exist_ok=True)
    pathlib.Path(negative_path_save_).mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(tqdm(os.listdir(positive_path_))):
        filepath_ = os.path.join(positive_path_, filename)

        img_ = cv2.imread(filepath_)
        img_orig_ = img_.copy()
        img_h_, img_w_ = img_.shape[:2]

        plate_mask_ = apply_grab_cut(img_)
        plate_mask_, biggest_contour_ = select_main_area_only(plate_mask_)

        # (x00, y00), r00 = cv2.minEnclosingCircle(biggest_contour)
        _, r00 = cv2.minEnclosingCircle(biggest_contour_)

        masked_img_ = cv2.bitwise_and(img_orig_, img_orig_, mask=plate_mask_)
        best_x_, best_y_, best_r_ = select_best_circle(masked_img_, plate_mask_)

        circle_mask_ = np.zeros(img_orig_.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask_, (best_x_, best_y_), int(round(best_r_)), 255, -1)
        masked_img_ = cv2.bitwise_and(masked_img_, masked_img_, mask=circle_mask_)

        size_length = np.sqrt(2) * best_r_

        img_up_left = masked_img_.copy()[
                      int(best_y_ - best_r_ / np.sqrt(2)): int(best_y_),
                      int(best_x_ - best_r_ / np.sqrt(2)): int(best_x_)
                      ]
        img_up_right = masked_img_.copy()[
                       int(best_y_ - best_r_ / np.sqrt(2)): int(best_y_),
                       int(best_x_): int(best_x_ + best_r_ / np.sqrt(2))
                       ]
        img_down_left = masked_img_.copy()[
                        int(best_y_): int(best_y_ + best_r_ / np.sqrt(2)),
                        int(best_x_ - best_r_ / np.sqrt(2)): int(best_x_)
                        ]
        img_down_right = masked_img_.copy()[
                         int(best_y_): int(best_y_ + best_r_ / np.sqrt(2)),
                         int(best_x_): int(best_x_ + best_r_ / np.sqrt(2))
                         ]

        cv2.imwrite(os.path.join(positive_path_save_, f'cleaned_{idx}_up_left.png'), img_up_left)
        cv2.imwrite(os.path.join(positive_path_save_, f'cleaned_{idx}_up_right.png'), img_up_right)
        cv2.imwrite(os.path.join(positive_path_save_, f'cleaned_{idx}_down_left.png'), img_down_left)
        cv2.imwrite(os.path.join(positive_path_save_, f'cleaned_{idx}_down_right.png'), img_down_right)

        cv2.imshow('original', img_orig_)
        cv2.imshow('result', masked_img_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # TODO Add center and a whole picture
