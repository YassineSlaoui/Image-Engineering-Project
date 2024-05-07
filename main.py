import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QAction, QFileDialog, \
    QMessageBox, QDialog, QSlider, QComboBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel("Please open an image to get started.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)
        self.save_button = None
        self.apply_button = None
        self.slider = None
        self.dialog = None
        self.transformed_image = None
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 800, 600)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('File')

        open_action = QAction('Open Image', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction('Save Image', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        transform_menu = self.menuBar().addMenu('Transformations')

        contrast_action = QAction('Change Contrast', self)
        contrast_action.triggered.connect(self.change_contrast)
        transform_menu.addAction(contrast_action)

        luminance_action = QAction('Change Luminance', self)
        luminance_action.triggered.connect(self.change_luminance)
        transform_menu.addAction(luminance_action)

        grayscale_action = QAction('Grayscale Conversion', self)
        grayscale_action.triggered.connect(self.grayscale_conversion)
        transform_menu.addAction(grayscale_action)

        filter_action = QAction('Apply Filter', self)
        filter_action.triggered.connect(self.apply_filter)
        transform_menu.addAction(filter_action)

        edge_detection_action = QAction('Edge Detection', self)
        edge_detection_action.triggered.connect(self.edge_detection)
        transform_menu.addAction(edge_detection_action)

        histogram_action = QAction('Display Histogram', self)
        histogram_action.triggered.connect(self.display_histogram)
        transform_menu.addAction(histogram_action)

        self.image = None

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_dialog.setViewMode(QFileDialog.List)
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(file_path)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            if len(self.image.shape) == 3:
                
                height, width, channel = self.image.shape
                bytes_per_line = 3 * width
                q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                
                height, width = self.image.shape
                bytes_per_line = width
                q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_img)
            self.image_label = QLabel()
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.setCentralWidget(self.image_label)
        else:
            self.image_label.setText("No image to display.")

    def apply_transformation(self):
        self.image = self.transformed_image
        self.transformed_image = None
        self.display_image()
        self.dialog.close()

    def save_image(self):
        if self.transformed_image is not None:
            file_path = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.jpg *.bmp)')[0]
            if file_path:  
                cv2.imwrite(file_path, self.transformed_image)
        elif self.image is not None:
            file_path = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.jpg *.bmp)')[0]
            if file_path:
                cv2.imwrite(file_path, self.image)
        else:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")

    def change_contrast(self, alpha=1.0):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")
            return

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Change Contrast')
        self.dialog.setGeometry(100, 100, 400, 200)

        self.slider = QSlider(Qt.Horizontal, self.dialog)
        self.slider.setMinimum(0)
        self.slider.setMaximum(30)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.on_change_contrast)

        self.apply_button = QPushButton('Apply', self.dialog)
        self.apply_button.clicked.connect(self.apply_transformation)

        self.save_button = QPushButton('Save', self.dialog)
        self.save_button.clicked.connect(self.save_image)

        self.image_label = QLabel(self.dialog)

        layout = QVBoxLayout(self.dialog)
        contrast_label = QLabel('Contrast:', self.dialog)
        layout.addWidget(contrast_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        self.dialog.setLayout(layout)

        self.transformed_image = None  
        self.on_change_contrast(10)  

        self.dialog.exec_()

    def on_change_contrast(self, alpha):
        alpha /= 10.0
        if len(self.image.shape) == 3:
            adjusted = np.clip(self.image * alpha, 0, 255).astype(np.uint8)
            height, width, channel = adjusted.shape
            bytes_per_line = 3 * width
            q_img = QImage(adjusted.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            adjusted = np.clip(self.image * alpha, 0, 255).astype(np.uint8)
            height, width = adjusted.shape
            bytes_per_line = width
            q_img = QImage(adjusted.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.transformed_image = adjusted

    def change_luminance(self, beta=50):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")
            return

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Change Luminance')
        self.dialog.setGeometry(100, 100, 400, 200)

        self.slider = QSlider(Qt.Horizontal, self.dialog)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_change_luminance)

        self.apply_button = QPushButton('Apply', self.dialog)
        self.apply_button.clicked.connect(self.apply_transformation)

        self.save_button = QPushButton('Save', self.dialog)
        self.save_button.clicked.connect(self.save_image)

        self.image_label = QLabel(self.dialog)

        layout = QVBoxLayout(self.dialog)
        luminance_label = QLabel('Luminance:', self.dialog)
        layout.addWidget(luminance_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        self.dialog.setLayout(layout)

        self.transformed_image = None  
        self.on_change_luminance(0)  

        self.dialog.exec_()

    def on_change_luminance(self, beta):
        if len(self.image.shape) == 3:  
            adjusted = cv2.convertScaleAbs(self.image, alpha=1.0, beta=beta)
            height, width, channel = adjusted.shape
            bytes_per_line = 3 * width
            q_img = QImage(adjusted.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:  
            adjusted = cv2.convertScaleAbs(self.image, alpha=1.0, beta=beta)
            height, width = adjusted.shape
            bytes_per_line = width
            q_img = QImage(adjusted.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.transformed_image = adjusted

    def apply_filter(self):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")
            return

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Apply Filter')
        self.dialog.setGeometry(100, 100, 400, 200)

        self.combo_box = QComboBox(self.dialog)
        self.combo_box.addItem("Average")
        self.combo_box.addItem("Median")
        self.combo_box.addItem("Min")
        self.combo_box.addItem("Max")
        self.combo_box.currentIndexChanged.connect(self.on_apply_filter)

        self.apply_button = QPushButton('Apply', self.dialog)
        self.apply_button.clicked.connect(self.apply_transformation)

        self.save_button = QPushButton('Save', self.dialog)
        self.save_button.clicked.connect(self.save_image)

        self.image_label = QLabel(self.dialog)

        layout = QVBoxLayout(self.dialog)
        filter_label = QLabel('Filter Type:', self.dialog)
        layout.addWidget(filter_label)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        self.dialog.setLayout(layout)

        self.transformed_image = None  
        self.on_apply_filter()  

        self.dialog.exec_()

    def on_apply_filter(self):
        filter_type = self.combo_box.currentText()
        if filter_type == "Average":
            kernel = np.ones((5, 5), np.float32) / 25
            filtered = cv2.filter2D(self.image, -1, kernel)
        elif filter_type == "Median":
            filtered = cv2.medianBlur(self.image, 5)
        elif filter_type == "Min":
            kernel = np.ones((5, 5), np.uint8)
            filtered = cv2.erode(self.image, kernel)
        elif filter_type == "Max":
            kernel = np.ones((5, 5), np.uint8)
            filtered = cv2.dilate(self.image, kernel)

        if len(self.image.shape) == 3:  
            height, width, channel = filtered.shape
            bytes_per_line = 3 * width
            q_img = QImage(filtered.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:  
            height, width = filtered.shape
            bytes_per_line = width
            q_img = QImage(filtered.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.transformed_image = filtered

    def grayscale_conversion(self):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")
            return

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Grayscale Conversion')
        self.dialog.setGeometry(100, 100, 400, 200)

        self.apply_button = QPushButton('Apply', self.dialog)
        self.apply_button.clicked.connect(self.apply_transformation)

        self.save_button = QPushButton('Save', self.dialog)
        self.save_button.clicked.connect(self.save_image)

        self.image_label = QLabel(self.dialog)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        self.dialog.setLayout(layout)

        if len(self.image.shape) == 3:  
            self.transformed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:  
            self.transformed_image = self.image.copy()  
        self.display_transformed_image()

        self.dialog.exec_()

    def edge_detection(self, threshold1=50, threshold2=150):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "You need to load an image first.")
            return

        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Edge Detection')
        self.dialog.setGeometry(100, 100, 400, 200)

        self.slider1 = QSlider(Qt.Horizontal, self.dialog)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(255)
        self.slider1.setValue(threshold1)
        self.slider1.valueChanged.connect(self.on_edge_detection)

        self.slider2 = QSlider(Qt.Horizontal, self.dialog)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(255)
        self.slider2.setValue(threshold2)
        self.slider2.valueChanged.connect(self.on_edge_detection)

        self.apply_button = QPushButton('Apply', self.dialog)
        self.apply_button.clicked.connect(self.apply_transformation)

        self.save_button = QPushButton('Save', self.dialog)
        self.save_button.clicked.connect(self.save_image)

        self.image_label = QLabel(self.dialog)

        layout = QVBoxLayout(self.dialog)
        threshold1_label = QLabel('Threshold 1:', self.dialog)
        layout.addWidget(threshold1_label)
        layout.addWidget(self.slider1)
        threshold2_label = QLabel('Threshold 2:', self.dialog)
        layout.addWidget(threshold2_label)
        layout.addWidget(self.slider2)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        self.dialog.setLayout(layout)

        self.on_edge_detection()

        self.dialog.exec_()

    def on_edge_detection(self):
        threshold1 = self.slider1.value()
        threshold2 = self.slider2.value()
        if len(self.image.shape) == 3:  
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:  
            gray = self.image.copy()  
        self.transformed_image = cv2.Canny(gray, threshold1, threshold2)
        self.display_transformed_image()

    def display_histogram(self):
        if self.image is not None:
            if len(self.image.shape) == 3:
                
                color = ('Blue', 'Green', 'Red')
                fig, axs = plt.subplots(2, 2, figsize=(12, 8))

                
                for i, col in enumerate(color):
                    row = i // 2
                    col = i % 2
                    histr = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    axs[row, col].plot(histr, color=color[i].lower())
                    axs[row, col].set_title(f'{color[i]} Channel')
                    axs[row, col].set_xlim([0, 256])

                
                ax = axs[1, 1]
                for i, col in enumerate(color):
                    histr = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    ax.plot(histr, color=color[i].lower(), label=f'{color[i]} Channel')
                ax.set_title('Superimposed Histograms')
                ax.set_xlim([0, 256])
                ax.legend()

                plt.tight_layout()
            else:
                
                
                hist = cv2.calcHist([self.image], [0], None, [256], [0, 256])

                
                fig, ax = plt.subplots()
                ax.plot(hist)
                ax.set_title('Histogram')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')

            
            self.dialog = QDialog(self)
            self.dialog.setWindowTitle('Histogram')
            self.dialog.setGeometry(100, 100, 800, 600)

            
            layout = QVBoxLayout()

            
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)

            
            toolbar = NavigationToolbar(canvas, self.dialog)
            layout.addWidget(toolbar)

            self.dialog.setLayout(layout)
            self.dialog.exec_()
        else:
            QMessageBox.warning(self, "Warning", "No image loaded.")

    def display_transformed_image(self):
        if self.transformed_image is not None:
            height, width = self.transformed_image.shape
            bytes_per_line = width
            q_img = QImage(self.transformed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("No image to display.")

    def print_image(self):
        print(self.image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
