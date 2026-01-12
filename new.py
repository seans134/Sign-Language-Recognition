import keras.models
import mediapipe as mp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# actions that we are detecting
actions = np.array(['hello', 'thanks', 'iloveyou'])

# path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_DATA')

# number of videos worth of data
no_sequence = 30

# videos 30 frames in length
sequence_length = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # flip horiztonally
    image = cv2.flip(image, 1)

    # Set flag to false
    image.flags.writeable = False

    # making detections
    results = model.process(image)

    # set flag to True
    image.flags.writeable = True

    # set RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=1)
                            )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(0, 0, 128), thickness=1, circle_radius=1)
                            )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

def video(model):
    sequence = []
    for n in range(30):
        sequence.append(np.zeros(1662))
    sentence = []
    threshold = 0.85

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)


            cv2.imshow('Feed', image)

            # checks if / has been pressed and breaks loop if it has
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def collect_data():
    #hello
    ## 0
    ## 1
    ## ...
    ## 29

    for action in actions:
        for sequence in range(no_sequence):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequence):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()

                    image, results = mediapipe_detection(frame, holistic)

                    draw_landmarks(image, results)

                    #wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'Starting Collection', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                    cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                    cv2.LINE_AA)

                    #export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    cv2.imshow('Feed', image)


                    # checks if / has been pressed and breaks loop if it has
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

def train():
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequence):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

    log_dir = os.path.join('logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    ###########fix es
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=500, callbacks=[tb_callback, es])

    model.save('action.h5')

    yhat = model.predict(x_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    accuracy = accuracy_score(ytrue, yhat)

def load_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('action.h5')
    return model

if __name__ == '__main__':
    model = load_model()
    video(model)
