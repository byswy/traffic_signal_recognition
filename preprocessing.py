import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def image_color_extract(img, lower, upper):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1 = cv2.inRange(img1, lower, upper)
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    img1 = cv2.dilate(img1, kernel)

    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    rects = []
    for contour in contours:
        a = cv2.contourArea(contour)
        if a < img.shape[0] * img.shape[1] / 256:
            continue
        # boxs = cv2.minAreaRect(contour)
        # points = cv2.boxPoints(boxs)
        # points = np.int0(points)
        # cv2.drawContours(img, [points], -1, (0, 255, 0), 2)

        x, y, w, h = cv2.boundingRect(contour)
        rects.append((x, y, w, h))
        # cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    img3 = np.zeros(shape=img.shape, dtype=np.uint8)
    for rect in rects:
        if rect[0] == 0 and rect[1] == 0:
            img3 = img1
        else:
            cv2.grabCut(img1, mask, rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img2 = img * mask2[:, :, np.newaxis]
            img3 = cv2.bitwise_or(img3, img2)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    t, img3 = cv2.threshold(img3, 1, 255, cv2.THRESH_BINARY)
    return img3


def image_process(img):
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    img_blue = image_color_extract(img, lower_blue, upper_blue)

    lower_red1 = np.array([156, 43, 46])
    upper_red1 = np.array([180, 255, 255])
    img_red1 = image_color_extract(img, lower_red1, upper_red1)

    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    img_red2 = image_color_extract(img, lower_red2, upper_red2)

    img0 = cv2.bitwise_or(img_blue, img_red1)
    img0 = cv2.bitwise_or(img0, img_red2)

    contours, hierarchy = cv2.findContours(img0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        img_cut = img
    else:
        x, y, w, h = cv2.boundingRect(contours[0])

        p1 = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        p2 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
        M = cv2.getPerspectiveTransform(p1, p2)
        img_cut = cv2.warpPerspective(img, M, (200, 200))

    return img_cut


if __name__ == '__main__':
    if not os.path.exists('./data/train_data'):
        os.makedirs('./data/train_data')

    root = './data/tsrd-train'
    for i in os.listdir(root):
        pat = os.path.join(root, i)
        p = os.path.join('./data/train_data', i)
        if not os.path.exists(p):
            os.makedirs(p)
        for j in os.listdir(pat):
            print(os.path.join(p, j))
            path = os.path.join(pat, j)
            img = cv2.imread(path)
            img0 = image_process(img)
            cv2.imwrite(os.path.join(p, j), img0)

    for i in os.listdir('./data/train_data'):
        path_dir = os.path.join('./data/train_data', i)
        lists = os.listdir(path_dir)
        print(i, len(lists), lists)
        # if len(lists) <= 10:
        #     for l in lists:
        #         path = os.path.join(path_dir, l)
        #         img = cv2.imread(path)
        #         img = cv2.resize(img, (48, 48))
        #         for k in range(30):
        #             ll = l + '_gai_' + str(k) + '.jpg'
        #             path = os.path.join(path_dir, ll)
        #             print(path)
        #             cv2.imwrite(path, img)
        # elif 10 < len(lists) <= 20:
        #     for l in lists:
        #         path = os.path.join(path_dir, l)
        #         img = cv2.imread(path)
        #         img = cv2.resize(img, (48, 48))
        #         for k in range(10):
        #             ll = l + '_gai_' + str(k) + '.jpg'
        #             path = os.path.join(path_dir, ll)
        #             print(path)
        #             cv2.imwrite(path, img)
        # elif 20 < len(lists) <= 30:
        #     for l in lists:
        #         path = os.path.join(path_dir, l)
        #         img = cv2.imread(path)
        #         img = cv2.resize(img, (48, 48))
        #         for k in range(5):
        #             ll = l + '_gai_' + str(k) + '.jpg'
        #             path = os.path.join(path_dir, ll)
        #             print(path)
        #             cv2.imwrite(path, img)
        # elif 30 < len(lists) <= 50:
        #     for l in lists:
        #         path = os.path.join(path_dir, l)
        #         img = cv2.imread(path)
        #         img = cv2.resize(img, (48, 48))
        #         for k in range(2):
        #             ll = l + '_gai_' + str(k) + '.jpg'
        #             path = os.path.join(path_dir, ll)
        #             print(path)
        #             cv2.imwrite(path, img)
