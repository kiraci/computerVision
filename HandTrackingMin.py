import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

currentTime = 0
previousTime = 0

while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    #print(results.multi_hand_landmarks)

    landmarks = results.multi_hand_landmarks

    if landmarks:
        for handLandmark in landmarks:
            for id, landmark in enumerate(handLandmark.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                # print(id, x, y)
                if id == 0:
                    cv2.circle(img, (x, y), 25, (255, 255, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    FPS = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(FPS)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    cv2.imshow("Main Camera", img)
    cv2.waitKey(1)