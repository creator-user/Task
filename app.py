# src/ui/app.py
import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from src.ui.design import Ui_MainWindow
from src.model.inference import preprocess_image, predict_label, load_trained_model
from src.config import BASE_DIR

# 设置类别和图像路径
LABELS = [
    "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"
]
IMAGE_DIR = {
    "Cardboard": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "cardboard"),
    "Glass": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "glass"),
    "Metal": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "metal"),
    "Paper": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "paper"),
    "Plastic": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "plastic"),
    "Trash": os.path.join(BASE_DIR, "data", "processed", "TrashNet", "train", "trash")
}


class MainWindow(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.predictButton.clicked.connect(self.predict_garbage)
        self.exitButton.clicked.connect(QtWidgets.QApplication.quit)
        self.previousButton.clicked.connect(self.previous_image)
        self.nextButton.clicked.connect(self.next_image)
        self.categoryListWidget.addItems(LABELS)
        self.categoryListWidget.currentItemChanged.connect(self.update_images)

        self.model = load_trained_model()
        self.current_images = []
        self.current_image_index = 0

    def update_images(self):
        """更新当前类别的图像列表"""
        category = self.categoryListWidget.currentItem().text()
        self.current_images = sorted(
            [f for f in os.listdir(IMAGE_DIR[category]) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        )
        self.current_image_index = 0
        self.display_image()

    def display_image(self):
        """在界面上显示当前图像"""
        if self.current_images:
            category = self.categoryListWidget.currentItem().text()
            image_path = os.path.join(IMAGE_DIR[category], self.current_images[self.current_image_index])
            pixmap = QtGui.QPixmap(image_path)
            pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setText("")
        else:
            self.imageLabel.setText("没有待显示的图像")

    def previous_image(self):
        """显示上一张图像"""
        if self.current_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.current_images)
            self.display_image()

    def next_image(self):
        """显示下一张图像"""
        if self.current_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.current_images)
            self.display_image()

    def predict_garbage(self):
        """预测当前图像的类别"""
        if self.current_images:
            category = self.categoryListWidget.currentItem().text()
            image_path = os.path.join(IMAGE_DIR[category], self.current_images[self.current_image_index])
            image_tensor = preprocess_image(image_path)
            prediction_label, confidence = predict_label(image_tensor, self.model)
            self.resultLabel.setText(f"识别结果: {prediction_label} ({confidence*100:.2f}% Confidence)")
        else:
            self.resultLabel.setText("无法识别")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
