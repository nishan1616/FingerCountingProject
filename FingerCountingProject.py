import cv2
import os
import mediapipe as mp
import time
import HandTrackingModule2 as htm

wcam,hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

folder_path = "FingerImages"
my_list = os.listdir(folder_path)
print(my_list)
overlay_list = []

for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    #print(f'{folder_path}/{img_path}')
    overlay_list.append(image)
print(len(overlay_list))
ptime = 0

detector = htm.HandDetector(detection_con=0.75)
tip_ids = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.draw_hands(img)
    landmarks_list = detector.find_landmarks(img,draw=False)
    #print(landmarks_list)
    if len(landmarks_list)!=0:
        fingers = []
        #Thumb
        if landmarks_list[4][1] > landmarks_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #4 fingers
        for id in range(1,5):
            if landmarks_list[tip_ids[id]][2] < landmarks_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)
        h,w,c = overlay_list[total_fingers-1].shape
        img[0:h, 0:w] = overlay_list[total_fingers-1]

        cv2.rectangle(img, (20,225),(175,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(total_fingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}',(450,40),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)