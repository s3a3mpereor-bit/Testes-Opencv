import cv2
import os

# Ler video
Video_path = os.path.join('.', 'Video', 'video.mp4')
video = cv2.VideoCapture(Video_path)

# Ver video
ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(40)

video.release()
cv2.destroyAllWindows()