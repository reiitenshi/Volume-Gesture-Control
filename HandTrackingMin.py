### Acknowledgement: Code learnt from Murtaza's Workshop on YouTube
### YouTube Link: https://www.youtube.com/watch?v=NZde8Xt78Iw

import cv2
import mediapipe as mp
import time # Shows FPS

# Basic setup to run a webcam
cap = cv2.VideoCapture(0)

# Hand Detection
mpHands = mp.solutions.hands        # Calling the model
hands = mpHands.Hands()             # Hands object with the model
mpDraw = mp.solutions.drawing_utils # Hand Landmarks map
# --

# FPS Setup
pTime = 0 # Previous
cTime = 0 # Current
# --

# Shows webcam
while True:
    success, img = cap.read()

    # Sends image to object hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) ## Shows detection results

    if results.multi_hand_landmarks:                                       # If Hand Landmarks are detected
        for handLms in results.multi_hand_landmarks:                       # For every Hand Landmarks that are detected
            for id, lm in enumerate(handLms.landmark):      # ID each landmark points with its coordinates
                # print(id, lm)                             ## Shows each Landmark's coordinates
                h, w, c = img.shape                     # Defining an image's shape with its coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)   # Turning the coordinates into Pixel locations
                print (id, cx, cy)                      # Shows Pixel locations of each Landmark

                # Highlight a certain point in the Landmark
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                # --

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # Draw its Hand Landmarks
    # --

    # FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # --

    cv2.imshow("Image", img)
    cv2.waitKey(1)