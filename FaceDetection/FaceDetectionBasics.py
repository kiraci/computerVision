import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

currentTime = 0
previousTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (123, 255, 14), 2)
            cv2.putText(img, f'Score: {int(detection.score[0] * 100)}', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (14, 80, 255), 2)

    currentTime = time.time()
    FPS = 1 / ( currentTime - previousTime )
    previousTime = currentTime
    cv2.putText(img, f'FPS: {int(FPS)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

    cv2.imshow("Face Camera", img)
    cv2.waitKey(1)