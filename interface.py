from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import cv2
import numpy as np
import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)  # 创建一个可以在给定图像上绘图的对象
    fontStyle = ImageFont.truetype('simsun.ttc', textSize, encoding='utf-8')  # 字体的格式
    draw.text(position, text, textColor, font=fontStyle)  # 绘制文本
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def image_predict(img):
    """
    图像预测
    :param img:
    :return:
    """
    img_, img_cuts, points = image_process(img)
    if len(img_cuts) == 0:
        return None, None
    else:
        model = tf.keras.models.load_model('./model/model_CNN.h5')
        # model.summary()

        preds = ''
        dic = {0: '步行', 1: '非机动车行驶', 2: '环岛行驶', 3: '机动车行驶', 4: '靠右侧道路行驶', 5: '靠左侧车道行驶',
               6: '立交直行和右转弯行驶', 7: '立交直行和左转弯行驶', 8: '鸣喇叭', 9: '向右转弯', 10: '向左和向右转弯',
               11: '向左转弯', 12: '直行', 13: '直行和向右转弯', 14: '直行和向左转弯', 15: '允许掉头', 16: '最低限速50',
               17: '人行横道', 18: '禁止超车', 19: '禁止农用运输车通行', 20: '禁止大型客车通行',
               21: '禁止直行和向左转弯', 22: '禁止掉头', 23: '禁止非机动车通行', 24: '禁止载货汽车通行',
               25: '禁止汽车拖、挂车通行', 26: '禁止行人通行', 27: '禁止机动车通行', 28: '禁止鸣喇叭',
               29: '禁止摩托车通行', 30: '禁止载货汽车、拖拉机通行', 31: '禁止直行', 32: '禁止人力车通行',
               33: '禁止人力货运三轮车通行', 34: '禁止人力客运三轮车通行', 35: '禁止三轮汽车、低速货车通行',
               36: '禁止向右转弯', 37: '禁止向左向右转弯', 38: '禁止直行和向右转弯', 39: '禁止拖拉机通行',
               40: '禁止向左转弯', 41: '禁止驶入', 42: '禁止车辆长时停放', 43: '禁止停车', 44: '限制宽度3m',
               45: '限制高度3.5m', 46: '限制质量10t', 47: '会车让行', 48: '禁止通行', 49: '停车检查', 50: '限制速度5',
               51: '限制速度15', 52: '限制速度30', 53: '限制速度40', 54: '限制速度50', 55: '限制速度60',
               56: '限制速度70', 57: '限制速度80', 58: '停车让行', 59: '减速让行'}
        for i in range(len(points)):
            point = points[i]
            img_cut = img_cuts[i]
            img0 = cv2.resize(img_cut, (48, 48))
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img0 = img0 / 255
            img0 = np.array([img0])
            p = model.predict(img0)
            pi = np.argmax(p)
            pi = dic[pi]
            preds += pi + '  '
            img_ = cv2AddChineseText(img_, pi, point, (0, 255, 0), 30)

        return img_, preds


def image_color_extract(img, lower, upper):
    """
    颜色提取
    :param img:
    :param lower:
    :param upper:
    :return:
    """
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
    """
    图像分割和处理
    :param img:
    :return:
    """
    # img = cv2.resize(img, (800, 800))

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    img_blue = image_color_extract(img, lower_blue, upper_blue)

    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])
    img_red1 = image_color_extract(img, lower_red1, upper_red1)

    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    img_red2 = image_color_extract(img, lower_red2, upper_red2)

    img0 = cv2.bitwise_or(img_blue, img_red1)
    img0 = cv2.bitwise_or(img0, img_red2)

    contours, hierarchy = cv2.findContours(img0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_cuts = []
    points = []
    for contour in contours:
        a = cv2.contourArea(contour)
        if a < img.shape[0] * img.shape[1] / 256:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.drawContours(img, [points], 0, (0, 255, 0), 2)

        p1 = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        p2 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])
        M = cv2.getPerspectiveTransform(p1, p2)
        img_cuts.append(cv2.warpPerspective(img, M, (200, 200)))
        points.append([x + w, y + h])

    # cv2.imshow('0', img0)
    # cv2.imshow('00', img)
    # for i in range(len(img_cuts)):
    #     cv2.imshow(str(i + 1), img_cuts[i])
    # cv2.waitKey(60000)

    return img, img_cuts, points


class interface(QWidget):
    def __init__(self):
        self.lab_picture = None
        self.lab_text = None
        super(interface, self).__init__()
        self.main_UI()
        self.button_UI()
        self.lab_text_UI('请导入图片！')

    def main_UI(self):
        self.setWindowTitle('interface')  # 设置窗口名称
        self.setGeometry(100, 100, 1200, 800)  # 设置窗位置和大小

    def button_UI(self):
        btn_1 = QPushButton('退出', self)
        btn_2 = QPushButton('添加并识别图片', self)

        # 设置位置和大小
        btn_1.setGeometry(1160, 0, 40, 800)
        btn_2.setGeometry(580, 760, 580, 40)

        # 按钮链接函数
        btn_1.clicked.connect(self.close)
        btn_2.clicked.connect(self.Tow)

    def lab_text_UI(self, text):
        if self.lab_text != None:
            self.lab_text.clear()
        self.lab_text = QLabel(text, self)
        self.lab_text.setWordWrap(True)
        self.lab_text.setFont(QFont('Arial', 10, QFont.Weight.Bold))  # 设置字体
        self.lab_text.setGeometry(0, 760, 580, 40)  # 设置位置和大小
        self.lab_text.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 水平方向对齐
        self.lab_text.show()

    def lab_picture_UI(self, x, y, a, b):
        if self.lab_picture != None:
            self.lab_picture.clear()
        self.lab_picture = QLabel(self)
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        a1, b1, c1 = img.shape
        c = c1 * b1
        img = QImage(img.data, b1, a1, c, QImage.Format.Format_RGB888)
        self.lab_picture.setPixmap(QPixmap.fromImage(img))  # 在label上显示图片
        self.lab_picture.setScaledContents(True)  # 让图片自适应label大小
        self.lab_picture.setGeometry(x, y, a, b)
        self.lab_picture.show()

    def Tow(self):
        # 设置文件扩展名过滤,用双分号间隔
        filepath, filetype = QFileDialog.getOpenFileName(self, '选取图片', './image/',
                                                         'All Files (*);;PNG (*.png);;JPG (*.jpg;*.jpeg;*.jpe;*.jfif)')
        # print(filepath)
        if filepath != '':
            self.img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), 1)
            shap = self.img.shape
            if shap[0] > 700:
                self.img = cv2.resize(self.img, (round(700 / shap[0] * shap[1]), 700))
                shap = self.img.shape
            if shap[1] > 1100:
                self.img = cv2.resize(self.img, (1100, round(1100 / shap[1] * shap[0])))
                shap = self.img.shape
            self.img, a = image_predict(self.img)
            if a == None:
                self.lab_text_UI('识别不到交通标识牌！')
                if self.lab_picture != None:
                    self.lab_picture.clear()
            else:
                self.lab_text_UI('预测结果为：' + a)
                self.lab_picture_UI(580 - shap[1] // 2, 380 - shap[0] // 2, shap[1], shap[0])

    def closeEvent(self, event):
        ok = QPushButton()
        cancel = QPushButton()
        msg = QMessageBox(QMessageBox.Icon.Warning, '关闭', '是否关闭！')
        msg.addButton(ok, QMessageBox.ButtonRole.ActionRole)
        msg.addButton(cancel, QMessageBox.ButtonRole.RejectRole)
        ok.setText('确定')
        cancel.setText('取消')
        if msg.exec() == 1:
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = interface()
    win.show()
    sys.exit(app.exec())
