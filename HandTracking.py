import cv2
import mediapipe as mp
import time
import HandTrackingModule as hndTrc

cap = cv2.VideoCapture(0)

currentTime = 0
previousTime = 0

detector = hndTrc.HandDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    landmarks = detector.findPosition(img)
    if len(landmarks) != 0:
        print(landmarks[4])

    currentTime = time.time()
    FPS = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    cv2.imshow("Main Camera", img)
    cv2.waitKey(1)