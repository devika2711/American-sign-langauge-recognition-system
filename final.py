import numpy as np
import cv2 as cv
import mediapipe as mp
from tensorflow.keras.models import load_model
from Function import *

# Load the CNN model
cnn_model = load_model('best_sign_language_model.h5')

holy_hands = mp.solutions.hands
cap = cv.VideoCapture(0)

def preprocess_for_cnn(image, bbox, size=(28, 28)):
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)

    cropped_image = image[y_min:y_max, x_min:x_max]
    gray_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
    resized_image = cv.resize(gray_image, size)
    normalized_image = resized_image / 255.0
    reshaped_image = normalized_image.reshape(1, size[0], size[1], 1)
    return reshaped_image

with holy_hands.Hands(max_num_hands=1) as hands:
    index_cord = []
    string = ''  # Initialize string outside the loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_cordinate = []
                imgH, imgW = image.shape[:2]
                x_max, y_max, x_min, y_min = 0, 0, imgW, imgH
                for index, landmark in enumerate(hand_landmarks.landmark):
                    x_cordinate, y_cordinate = int(landmark.x * imgW), int(landmark.y * imgH)
                    hand_cordinate.append([index, x_cordinate, y_cordinate])
                    x_max, y_max = max(x_max, x_cordinate), max(y_max, y_cordinate)
                    x_min, y_min = min(x_min, x_cordinate), min(y_min, y_cordinate)
                hand_cordinate = np.array(hand_cordinate)
                bbox = [x_min, y_min, x_max, y_max]

                cnn_input = preprocess_for_cnn(image, bbox)
                cnn_prediction = cnn_model.predict(cnn_input)

                string = persons_input(hand_cordinate)
                image = get_fram(image, hand_cordinate, string)

        # Pointer logic
        if string == "D":
            index_cord.append([15, hand_cordinate[8][1], hand_cordinate[8][2]])
        elif string in ["I", "J"]:
            index_cord.append([15, hand_cordinate[20][1], hand_cordinate[20][2]])

        for val in index_cord:
            image = cv.circle(image, (val[1], val[2]), val[0], (255, 255, 255), 1)
            val[0] -= 1
            if val[0] <= 0:
                index_cord.remove(val)

        cv.imshow('Sign Language detection', cv.flip(image, 1))

        if cv.waitKey(5) & 0xFF == ord('x'):
            break

cap.release()
cv.destroyAllWindows()
