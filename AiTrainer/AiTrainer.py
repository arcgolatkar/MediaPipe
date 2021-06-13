import cv2
import time
import numpy as np
import PoseModule as pm

cap = cv2.VideoCapture("pushups_1.mp4")
cap.set(3, 640)
cap.set(4, 480)
ptime = 0
detector = pm.poseDetector()
count = 0
dir = 0 #0 when going up and 1 for down

while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    img = cv2.resize(img,(1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15, False)
        per = np.interp(angle, (195, 310), (0, 100))
        bar = np.interp(angle, (195, 310), (650, 100))
        #print(angle, per)

        # check pushup
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count +=0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count +=0.5
                dir = 0
        print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 0, 0), 25)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("VideoStream", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
