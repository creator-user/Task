# src/data_processing/preprocess.py
import cv2 as cv
import numpy as np


def load_image(filepath):
    return cv.imread(filepath)


def resize_image(image, width, height):
    return cv.resize(image, (width, height))


def normalize_image(image):
    return image / 255.0


def augment_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(image, M, (image.shape[1], image.shape[0]))


def save_image(image, path):
    cv.imwrite(path, image)


def crop_image(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def extract_contours(binary_image):
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def binarize_image(image, threshold=127):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    return binary
