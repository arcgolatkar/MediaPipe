import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

ctime = 0
ptime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img) #draw can be disabled
    lmList = detector.findPosition(img) #draw can be disabled
    if len(lmList) != 0:
        print(lmList[4])
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    #img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break