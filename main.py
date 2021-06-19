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

        for bg_rect in background_rectangles:
            cv2.rectangle(mask, bg_rect[:2], bg_rect[2:], cv2.GC_BGD, -1)

        cv2.circle(mask, (int(round(img_w / 2)), int(round(img_h / 2))), FOREGROUND_RADIUS, cv2.GC_FGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        new_mask, new_bgd_model, new_fgd_model = cv2.grabCut(
            img_rotated, mask, None, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_MASK  # TODO iter cnt
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


def select_best_circle(masked_img, mask, biggest_contour):
    img_h, img_w = masked_img.shape[:2]

    moments = cv2.moments(mask)
    mean_x = int(round(moments['m10'] / moments['m00']))
    mean_y = int(round(moments['m01'] / moments['m00']))

    _, r00 = cv2.minEnclosingCircle(biggest_contour)

    masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        masked_img_gray, cv2.HOUGH_GRADIENT, 1, minDist=10,
        param1=20, param2=20,
        minRadius=int(r00 // 2), maxRadius=int(r00)
    )

    def emptiness_percentage(circle_xyr):
        center_x, center_y, radius = circle_xyr
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
                abs(c[0] - mean_x) <= 0.1 * img_w and abs(c[1] - mean_y) <= 0.1 * img_h
                or abs(c[0] - img_w / 2) <= 0.1 * img_w and abs(c[1] - img_h / 2) <= 0.1 * img_h,
                circles[0]
            )
        )
        circles_emptiness_ratio = list(map(emptiness_percentage, filtered_circles))
        circle_info = list(zip(filtered_circles, circles_emptiness_ratio))
        filtered_circle_info = list(filter(lambda circle_xyr_d: circle_xyr_d[1] < EMPTINESS_THRESHOLD, circle_info))
        if len(filtered_circle_info) == 0:
            best_x, best_y, best_r = img_w // 2, img_h // 2, FOREGROUND_RADIUS
        else:
            best_circle_info = max(filtered_circle_info, key=lambda c: c[0][2])
            best_x, best_y, best_r = best_circle_info[0]
    else:
        best_x, best_y, best_r = img_w // 2, img_h // 2, FOREGROUND_RADIUS

    return best_x, best_y, best_r


def process_img(img):
    img_orig = img.copy()

    plate_mask = apply_grab_cut(img)
    plate_mask, biggest_contour = select_main_area_only(plate_mask)

    masked_img = cv2.bitwise_and(img_orig, img_orig, mask=plate_mask)
    best_x, best_y, best_r = select_best_circle(masked_img, plate_mask, biggest_contour)

    circle_mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
    cv2.circle(circle_mask, (best_x, best_y), int(round(best_r)), 255, -1)
    masked_img = cv2.bitwise_and(masked_img, masked_img, mask=circle_mask)

    img_center = masked_img.copy()[
                 int(best_y - best_r / np.sqrt(2)): int(best_y + best_r / np.sqrt(2)),
                 int(best_x - best_r / np.sqrt(2)): int(best_x + best_r / np.sqrt(2))
                 ]

    img_up_left = masked_img.copy()[
                  int(best_y - best_r / np.sqrt(2)): int(best_y),
                  int(best_x - best_r / np.sqrt(2)): int(best_x)
                  ]
    img_up_right = masked_img.copy()[
                   int(best_y - best_r / np.sqrt(2)): int(best_y),
                   int(best_x): int(best_x + best_r / np.sqrt(2))
                   ]
    img_down_left = masked_img.copy()[
                    int(best_y): int(best_y + best_r / np.sqrt(2)),
                    int(best_x - best_r / np.sqrt(2)): int(best_x)
                    ]
    img_down_right = masked_img.copy()[
                     int(best_y): int(best_y + best_r / np.sqrt(2)),
                     int(best_x): int(best_x + best_r / np.sqrt(2))
                     ]

    return img_center, img_up_left, img_up_right, img_down_left, img_down_right


if __name__ == '__main__':
    positive_path_ = r'./data/train/cleaned'
    positive_path_save_ = r'data/train/processed_train/cleaned'
    negative_path_ = r'./data/train/dirty'
    negative_path_save_ = r'data/train/processed_train/dirty'
    test_path_ = r'./data/test'
    test_path_save_ = r'./data/processed_test'

    pathlib.Path(positive_path_save_).mkdir(parents=True, exist_ok=True)
    pathlib.Path(negative_path_save_).mkdir(parents=True, exist_ok=True)

    process_type_ = 'test'  # 'cleaned', 'dirty', 'test'

    # data_path = r'./data/test'
    # path_to_save = r'./data/processed_test'
    # for filename in os.listdir(data_path):
    #     filepath = os.path.join(data_path, filename)
    #     save_prefix = filename.split('.')[0]

    if process_type_ == 'cleaned':
        data_path_ = positive_path_
        save_path_ = positive_path_save_
    elif process_type_ == 'dirty':
        data_path_ = negative_path_
        save_path_ = negative_path_save_
    else:
        data_path_ = test_path_
        save_path_ = test_path_save_

    for idx_, filename_ in enumerate(tqdm(os.listdir(data_path_))):
        filepath_ = os.path.join(data_path_, filename_)
        img_ = cv2.imread(filepath_)
        img_center_, img_up_left_, img_up_right_, img_down_left_, img_down_right_ = process_img(img_)

        cv2.imwrite(os.path.join(save_path_, f'{process_type_}_{filename_.split(".")[0]}_{idx_}_up_left.png'),
                    img_up_left_)
        cv2.imwrite(os.path.join(save_path_, f'{process_type_}_{filename_.split(".")[0]}_{idx_}_up_right.png'),
                    img_up_right_)
        cv2.imwrite(os.path.join(save_path_, f'{process_type_}_{filename_.split(".")[0]}_{idx_}_down_left.png'),
                    img_down_left_)
        cv2.imwrite(os.path.join(save_path_, f'{process_type_}_{filename_.split(".")[0]}_{idx_}_down_right.png'),
                    img_down_right_)
        cv2.imwrite(os.path.join(save_path_, f'{process_type_}_{filename_.split(".")[0]}_{idx_}_center.png'),
                    img_center_)
