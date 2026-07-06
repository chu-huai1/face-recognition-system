import cv2
import numpy as np

def retinex_enhance(image, sigma=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    log_img = np.log1p(gray.astype(np.float32))
    blur = cv2.GaussianBlur(log_img, (0, 0), sigma)
    log_reflect = log_img - blur
    reflect = np.expm1(log_reflect)
    reflect = cv2.normalize(reflect, None, 0, 255, cv2.NORM_MINMAX)
    reflect = reflect.astype(np.uint8)
    return cv2.cvtColor(reflect, cv2.COLOR_GRAY2BGR)
