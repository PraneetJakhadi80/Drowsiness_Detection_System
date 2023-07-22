import cv2
import numpy as np
import winsound
from tensorflow.keras.models import load_model

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained eye model
eye_model = load_model(r"C:\Users\lenovo\Desktop\Eye Detection\model.h5")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FPS, 5)
counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            print("Eyes are not detected")
        else:
            for (ex, ey, ew, eh) in eyes:
                eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]

                # Preprocess the eye region for prediction
                final_image = cv2.resize(eyes_roi, (64,64))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0
                

                # Make predictions using the eye model
                predictions = eye_model.predict(final_image)

                if predictions >= 0.3:
                    status = "Open Eyes"
                    cv2.putText(frame, status, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_4)
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                    cv2.putText(frame, 'Active', (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    counter = counter + 1
                    status = "Closed Eyes"
                    cv2.putText(frame, status, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_4)
                    x1, y1, w1, h1 = 0, 0, 175, 75
                    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

                    if counter > 10:
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                        cv2.putText(frame, "Sleep Alert !!!", (x1 + int(w1 / 10), y1 + int(h1 / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        winsound.Beep(1000, 500)
                        counter = 0

    cv2.imshow("sleepy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


