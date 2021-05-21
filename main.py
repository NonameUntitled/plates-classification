import os
import cv2
import numpy as np

mainRectSize = 0.04
fgSize = 0.04

for filename in list(os.listdir(r'D:\University\Kaggle\plates-classification\data\train\cleaned')):
    filepath = os.path.join(r'D:\University\Kaggle\plates-classification\data\train\cleaned', filename)

    img = cv2.imread(filepath)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig = img.copy()
    img_h, img_w = img.shape[:2]
    mask = np.ones((img_h, img_w), dtype=np.uint8)

    bg_w = round(img_w * mainRectSize)
    bg_h = round(img_h * mainRectSize)
    bg_rect = (bg_w, bg_h, img_w - 2 * bg_w, img_h - 2 * bg_h)

    fg_w = round(img_w * (1 - fgSize) / 2)
    fg_h = round(img_h * (1 - fgSize) / 2)
    fg_rect = (fg_w, fg_h, img_w - fg_w, img_h - fg_h)

    cv2.rectangle(mask, fg_rect[:2], fg_rect[2:], color=cv2.GC_FGD, thickness=-1)

    bgd_model = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, bg_rect, bgd_model, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    cv2.rectangle(
        mask,
        (bg_w, bg_h),
        (img_w - bg_w, img_h - bg_h),
        color=cv2.GC_PR_BGD,
        thickness=bg_w * 3
    )

    cv2.grabCut(img, mask, bg_rect, bgd_model, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask_new = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    contours, _ = cv2.findContours(mask_new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(contours, key=cv2.contourArea)
    (x00, y00), r00 = cv2.minEnclosingCircle(biggest_contour)

    for y in range(mask_new.shape[0]):
        for x in range(mask_new.shape[1]):
            result = cv2.pointPolygonTest(biggest_contour, (x, y), False)
            if result >= 0:
                mask_new[y, x] = 255
            else:
                mask_new[y, x] = 0

    moments = cv2.moments(mask_new)

    mean_x = int(round(moments['m10'] / moments['m00']))
    mean_y = int(round(moments['m01'] / moments['m00']))

    # img = cv2.circle(img, (mean_x, mean_y), 4, (255, 0, 0), 5)
    # img = cv2.circle(img, (int(x00), int(y00)), int(r00), (0, 255, 0), 5)
    # img_orig = cv2.circle(img_orig, (int(x00), int(y00)), int(r00), (0, 255, 0), 5)

    # cv2.imshow('pre-changed', masked)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    masked = cv2.bitwise_and(img, img, mask=mask_new)
    masked[mask_new == 0] = [0, 0, 0]  # TODO 255

    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        masked_gray, cv2.HOUGH_GRADIENT, 1, 3,
        param1=5, param2=5,
        minRadius=int(r00 // 2), maxRadius=int(r00))

    if circles is not None:
        best_circle = max(
            list(filter(lambda c: abs(c[0] - x00) <= 0.1 * img_w and abs(c[1] - y00) <= 0.1 * img_h, circles[0])),
            key=lambda c: c[2]
        )
        best_x, best_y, best_r = best_circle
        img_orig = cv2.circle(img_orig, (int(best_x), int(best_y)), int(best_r), (0, 0, 255), 2)
    else:
        pass  # TODO

    size_length = np.sqrt(2) * r00  # TODO try other options

    img_up_left = masked.copy()[
                  int(y00 - r00 / np.sqrt(2)): int(y00),
                  int(x00 - r00 / np.sqrt(2)): int(x00)
                  ]
    img_up_right = masked.copy()[
                  int(y00 - r00 / np.sqrt(2)): int(y00),
                  int(x00): int(x00 + r00 / np.sqrt(2))
                  ]
    img_down_left = masked.copy()[
                  int(y00): int(y00 + r00 / np.sqrt(2)),
                  int(x00 - r00 / np.sqrt(2)): int(x00)
                  ]
    img_down_right = masked.copy()[
                  int(y00): int(y00 + r00 / np.sqrt(2)),
                  int(x00): int(x00 + r00 / np.sqrt(2))
                  ]

    # TODO try this also
    # img_cropped = img_orig[
    #               int(best_y - best_r / np.sqrt(2)): int(best_y + best_r / np.sqrt(2)),
    #               int(best_x - best_r / np.sqrt(2)): int(best_x + best_r / np.sqrt(2))
    #               ]
    # img_cropped = masked.copy()[
    #               int(y00 - r00 / np.sqrt(2)): int(y00 + r00 / np.sqrt(2)),
    #               int(x00 - r00 / np.sqrt(2)): int(x00 + r00 / np.sqrt(2))
    #               ]

    cv2.imshow('origin', img_orig)
    cv2.imshow('changed', masked)
    cv2.imshow('left_up', img_up_left)
    cv2.imshow('right_up', img_up_right)
    cv2.imshow('left_down', img_down_left)
    cv2.imshow('right_down', img_down_right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
