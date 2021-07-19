import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        #print(results.multi_hand_landmarks)

        landmarks = self.results.multi_hand_landmarks

        if landmarks:
            for handLandmark in landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarks = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                # print(id, x, y)
                landmarks.append([id, x, y])
                if draw:
                    cv2.circle(img, (x, y), 17, (123, 76, 17), cv2.FILLED)

        return landmarks

def main():
    cap = cv2.VideoCapture(0)

    currentTime = 0
    previousTime = 0

    detector = HandDetector()

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


if __name__ == "__main__":
    main()