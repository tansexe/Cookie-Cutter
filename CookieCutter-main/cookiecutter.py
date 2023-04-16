import cv2 as cv
from cvzone import HandTrackingModule, overlayPNG
import numpy as np
import os
import mediapipe as mp 
import math 
from skimage.measure import compare_ssim

folderPath = 'framesC'
mylist = os.listdir(folderPath)
graphic = [cv.imread(f'{folderPath}/{imPath}') for imPath in mylist] 
intro = graphic[0]
kill = graphic[1]
winner = graphic[2] 
cam = cv.VideoCapture(0) 
detector = HandTrackingModule.HandDetector(maxHands=1,detectionCon=0.77)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 

cv.imshow('Dalgona Game', cv.resize(intro, (0, 0), fx=0.69, fy=0.69))
cv.waitKey(1)

sqr_img = cv.imread('sqr(2).png')
mlsa =  cv.imread('mlsa.png')

#INTRO SCREEN WILL STAY UNTIL Q IS PRESSED
while True:
    cv.imshow('Dalgona Game', cv.resize(intro, (0, 0), fx=0.69, fy=0.69))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

TIMER_MAX = 45
TIMER = TIMER_MAX
maxMove = 10
font = cv.FONT_HERSHEY_SIMPLEX
cap = cv.VideoCapture(0)
frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH) 


gameOver = False
NotWon =True

capture = cv.VideoCapture(0)
detector = HandTrackingModule.HandDetector(maxHands=2, detectionCon=0.77)
## max hands for no of hands we need to detect
## detectionCon for percentage of error we can allow. Range is from 0 to 1

while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame, flipType=True)

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

    cv.imshow('Video', frame)


    if(cv.waitKey(20) & 0xFF==ord('q')):
        break
    
img_path = input("Enter the path of the image: ")
img = cv.imread(img_path)


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


edges = cv.Canny(gray, 100, 200)


img_with_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
img_with_edges[:,:,0] = 0 # set blue channel to 0
img_with_edges[:,:,1] = 0 # set green channel to 0
img_with_edges[:,:,2] = 255 # set red channel to 255

traced_image = cv.imread("traced_image.jpg")
original_image = cv.imread("original_image.jpg")


traced_image_gray = cv.cvtColor(traced_image, cv.COLOR_BGR2GRAY)
original_image_gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


ssim_value = compare_ssim(traced_image_gray, original_image_gray)

threshold = 0.95

if ssim_value >= threshold:
    print("You Win!")
else:
    print("You Lose!")

cv.imshow("Original Image", img)
cv.imshow("Traced Edges", img_with_edges)
cv.waitKey(0)
capture.release()
cv.destroyAllWindows() 