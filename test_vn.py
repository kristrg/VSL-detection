###########################
### IMPORT DEPENDENCIES ###
###########################
from typing import Sequence
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(["xin_chao","tam_biet","cam_on","vui","khoe","bo","me","toi","ban","gap","ten","kcg","co","khong","i_love_you","ong","hoc","vuii","ban_be","sach","doc"]) # Actions that we try to detect
no_sequences = 100                                  # Number of videos that we want to collect for our dataset
sequence_length = 30                                # Number of frames per video

model_path = "model_training/MODEL_LSTM_5.h5"
model = tf.keras.models.load_model(model_path)
########################
### DEFINE FUNCTIONS ###
########################
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion BGR to RGB
    image.flags.writeable = False                  # image is not writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color conversion RGB to BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)      # draw the connection map of face landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)          # draw the connection map of pose landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)     # draw the connection map of left-hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)    # draw the connection map of right-hand landmarks


def draw_landmarks_styled(image, results):
    # draw the connection map of face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                    mp_drawing.DrawingSpec(color=(80,120,10), thickness=1, circle_radius=1),    # dot color
                                    mp_drawing.DrawingSpec(color=(80,255,120), thickness=1, circle_radius=1))   # line color
    
    # draw the connection map of pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(120,30,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(120,70,110), thickness=2, circle_radius=2))
    
    # draw the connection map of left-hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(122,160,255), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(114,128,250), thickness=2, circle_radius=2))
    
    # draw the connection map of right-hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(154,250,0), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(87,139,46), thickness=2, circle_radius=2))


def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lefthand,righthand])


def prob_visualization(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100),90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

####################
### MAIN PROCESS ###
####################
sequence = []
sentence = []
predictions = []
threshold = 0.8
# colors = [(245,117,16), (117,245,16), (16,117,245)]

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read frame
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        # draw_landmarks_styled(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

        # Visualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res) and np.argmax(res)!=11:
                if res[np.argmax(res)] > threshold:
                    
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # image = prob_visualization(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640,40), (120,70,110), -1)
        # print(sentence[-5:])
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        # Show
        cv2.imshow("MediaPipe Holistic", image)

        # Break
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()