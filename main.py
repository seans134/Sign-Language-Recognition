import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import csv

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#def export_landmarks(results, character, hand, filename="landmarks.csv"):
    #with open('output.csv', 'a', newline='') as csvfile:
        #writer = csv.writer(csvfile)

d = {}
def add_list(results, character):
    d[character] = results


def video():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            #set it to BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #flip horiztonally
            image = cv2.flip(image, 1)

            #Set flag to false
            image.flags.writeable = False

            #making detections
            results = hands.process(image)

            #set flag to True
            image.flags.writeable = True

            #set RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #print(dir(results))

            #rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=2, circle_radius=2)
                                            )

            cv2.imshow('Hand Tracking', image)

            #checks if q has been pressed and breaks loop if it has

            c = cv2.waitKey(1)
            if c == 47:
                break
            elif c == 97:
                add_list(results, 'a')

            # checks if / has been pressed and breaks loop if it has
            #if cv2.waitKey(1) & 0xFF == ord('/'):
                #break

    cap.release()
    cv2.destroyAllWindows()

video()

a = {}
for i in d.keys():
    for landmark in d[i].multi_hand_landmarks:
        l = []
        for num in range(0, 21):
            x = landmark.landmark[num].x
            y = landmark.landmark[num].y
            z = landmark.landmark[num].z
            l.append([x,y,z])
        n = enumerate(l)
        
for i in n:
    print(i)

