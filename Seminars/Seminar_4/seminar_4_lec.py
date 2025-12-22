import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("Seminars/Seminar_4/data/video_cat.mp4")
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# slide 8 in lection_4
while True:
    ret, frame = cap.read()
    if not ret: break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    It = cv2.absdiff(gray, prev_gray)
    cv2.imshow("Frame", gray.view())
    cv2.imshow("Temporal difference |It|", It)
    # jcv2.imshow("Frame", gray.view(), colorspace='gray')
    # jcv2.imshow("Temporal difference |It|", It, colorspace='gray')
    
    prev_gray = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# cv2.waitKey(0)
# cv2.destroyAllWindows()