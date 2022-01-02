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
import streamlit as st
from PIL import Image
import tempfile


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # for the purpose of testing only

actions = np.array(["xin_chao","tam_biet","cam_on","vui","khoe","bo","me","toi","ban","gap","ten","kcg","co","khong","i_love_you","ong","hoc","vuii","ban_be","sach","doc"]) # Actions that we try to detect
no_sequences = 100                                  # Number of videos that we want to collect for our dataset
sequence_length = 30                                # Number of frames per video

model_path = "model_training/MODEL_LSTM_5.h5"
model = tf.keras.models.load_model(model_path)

###########################
######## MAIN APP #########
###########################
st.title("HANDSPEAK - Vietnamese Sign Language Recognition with MediaPipe")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("HandSpeak Sidebar")


def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # color conversion BGR to RGB
    image.flags.writeable = False                  # image is not writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color conversion RGB to BGR
    return image, results


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


menu = ["Quiz", "Demo", "About MediaPipe"]
app_mode = st.sidebar.selectbox("Choose the App mode", menu)

if app_mode == "Quiz":
    st.markdown("Let's get started with some quizzz <3")
    st.markdown("---")
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('''
                ### Question #1 \n
                **Sign Languague** is a system of communication using hands gesture to convey meaning. **True or False**?\n
    ''')
    st.text("Choose the answer below")
    if st.button("True"):
        st.success("CORRECT!")
        st.write("But hands gesture is not enough. Sign languages are expressed through manual articulations in combination with non-manual elements (hands gesture, **facial expression, body movement**, etc.)")
        st.balloons()
    if st.button("False"):
        st.error("INCORRECT!")
        st.write("You suree?!")

    st.markdown("---")
    st.markdown('''
                ### Question #2 \n
                **Sign Languague is universal** and Deaf people from around the world can communicate with each other by that universal sign language. **True or False**?\n
    ''')
    st.text("Choose the answer below")
    if st.button("Of course it is!"):
        st.error("INCORRECT!")
        st.markdown("**Did you really follow my presentation?!**")
    if st.button("No, it's not universal"):
        st.success("CORRECT!")
        st.markdown('''
                    Sign languages are not universal and they are not mutually intelligible with each other, although there are also striking similarities among sign languages.\n
                    According to Ethnologue Index, there are **103 sign languages** all over the world. 
                    ''')
        
        col1, col2 = st.columns(2)

        with col1:
            col1 = st.video("anh_em_MB.mp4")
            st.markdown('*Ngôn ngữ kí hiệu miền Bắc cho từ "anh em"*')

        with col2:
            col2 = st.video("anh_em_MN.mp4")
            st.markdown('*Ngôn ngữ kí hiệu miền Nam cho từ "anh em"*')

    st.markdown("---")
    st.markdown('''
                ### Question #3 \n
                When was Vietnamese Sign Language created?
    ''')
    st.text("Choose the answer below")
    if st.button("18th century"):
        st.error("INCORRECT!")
    if st.button("19th century"):
        st.success("CORRECT!")
    if st.button("20th century"):
        st.error("INCORRECT!")
    


if app_mode == "About MediaPipe":
    st.markdown("In this Application, **MediaPipe** is used for creating keypoints landmarks of the face, hands & pose. **StreamLit** is to create Web Graphical User Interface (GUI).")
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('''
                ### About MediaPipe \n
                MediaPipe is a cross-platform, customizable ML solutions for live and streaming media, developed by Google. \n
                You can check out more about MediaPipe via their Website & GitHub
                - [Website](https://google.github.io/mediapipe/)
                - [GitHub](https://github.com/google/mediapipe)
    ''')

    st.markdown("An introduction about MediaPipe")
    st.video("https://www.youtube.com/watch?v=YjYahrtFsM0")

if app_mode == "Demo":
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8

    # st.set_option("deprecation.showfileUploaderEncoding", False)
    use_webcam = st.sidebar.button("Use Webcam")
    draw_lm = st.sidebar.checkbox("Draw landmarks")

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("## Output")
    stframe = st.empty()

    # Get input video via webcam
    if use_webcam:
        cap = cv2.VideoCapture(0)

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
            while cap.isOpened():

                # Read frame
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                if draw_lm:
                    draw_landmarks_styled(image, results)
                else:
                    pass

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
                cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
                stframe.image(image, channels = "BGR", use_column_width=True)