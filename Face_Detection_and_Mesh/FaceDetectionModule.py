import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        #print(results.detections)
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img,detection)
                # print(id, detection)
                # print(results.score)
                #print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f' {int(detection.score[0]*100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255,0,255), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, len = 30, thic = 5, recthic = 1):
        x,y,w,h = bbox
        x1,y1 = x + w , y + h
        cv2.rectangle(img, bbox, (255, 0, 255), recthic)
        # Top Left x,y
        cv2.line(img, (x,y), (x+len,y),(255,0,255), thic)
        cv2.line(img, (x, y), (x , y+ len), (255, 0, 255), thic)
        # Top Right x,y
        cv2.line(img, (x1, y), (x1 - len, y), (255, 0, 255), thic)
        cv2.line(img, (x1, y), (x1, y + len), (255, 0, 255), thic)
        # Btm Left x,y
        cv2.line(img, (x, y1), (x + len, y1), (255, 0, 255), thic)
        cv2.line(img, (x, y1), (x, y1 - len), (255, 0, 255), thic)
        # Btm Right x,y
        cv2.line(img, (x1, y1), (x1 - len, y1), (255, 0, 255), thic)
        cv2.line(img, (x1, y1), (x1, y1 - len), (255, 0, 255), thic)
        return img


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()