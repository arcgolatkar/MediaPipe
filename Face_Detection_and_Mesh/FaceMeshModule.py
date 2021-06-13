import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode = False, maxFaces = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for id, faceLms in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                face = []
                for id1, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #print(id1, x, y)
                    # cv2.putText(img, str(id1), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5,
                    #             (0, 255, 0), 1)
                    face.append([id1, x, y])
                faces.append([id,face])
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img , faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(len(faces[0][1]))
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ =="__main__":
    main()
