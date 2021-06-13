import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import os

wCam, hCam = 640, 480
ptime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerPhotos"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imgPath in myList:
    img = cv2.imread(f'{folderPath}/{imgPath}')
    img = cv2.resize(img, (100, 250), interpolation=cv2.INTER_AREA)  # making constant size of images
    overlayList.append(img)

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []
        # thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]

        cv2.rectangle(img, (0, 300), (100, 600), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (30, 400), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    # img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    cv2.putText(img, str(int(fps)), (550, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
