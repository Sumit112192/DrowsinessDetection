import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained drowsiness detection model
model = load_model('best.keras')

def apply_gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def detect_and_crop_right_eye_from_webcam(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)

        if len(eyes) > 0:
            # Sort eyes by x coordinate to identify the right eye
            eyes = sorted(eyes, key=lambda e: e[0])
            (ex, ey, ew, eh) = eyes[0]  # Right eye assumed to have the smaller x coordinate
            
            # Crop the right eye region
            right_eye = frame[y + ey : y + ey + eh, x + ex : x + ex + ew]
        else:
            # If no eyes are detected, approximate the right eye region
            right_eye_x = x + int(w * 0.15) 
            right_eye_y = y + int(h * 0.3)
            right_eye_w, right_eye_h = int(w * 0.3), int(h * 0.2)

            right_eye = frame[right_eye_y : right_eye_y + right_eye_h, right_eye_x : right_eye_x + right_eye_w]

        # Resize the cropped right eye to 180x180
        right_eye = cv2.resize(right_eye, (180, 180))

        # Apply gamma correction to brighten the eye region
        # Gammar Correction is applied to the eyes to brighten up the image
        right_eye = apply_gamma_correction(right_eye, gamma=1.5)

        return right_eye

    return None

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frames
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect and crop the right eye
        right_eye = detect_and_crop_right_eye_from_webcam(frame)

        if right_eye is not None:
            gray_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
            gray_eye = np.expand_dims(gray_eye, axis=-1)
            gray_eye = np.expand_dims(gray_eye, axis=0)
            # Normalize the eye
            gray_eye = gray_eye / 255.0

            # Make the prediction using the drowsiness detection model
            prediction = model.predict(gray_eye)

            # Check if the model predicts drowsiness (based on your model's output)
            if prediction[0][0] > 0.5:
                cv2.putText(frame, "Drowsiness Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame with the detection and alert
        cv2.imshow('Drowsiness Detection', frame)

        # Exit the webcam feed if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

