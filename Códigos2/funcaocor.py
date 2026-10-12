import cv2
import numpy as np

def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    limiteinferior = hsvC[0][0][0] - 10, 100, 100
    limitesuperior = hsvC[0][0][0] + 10, 255, 255

    limiteinferior = np.array(limiteinferior, dtype=np.uint8)
    limitesuperior = np.array(limitesuperior, dtype=np.uint8)

    return limiteinferior, limitesuperior
