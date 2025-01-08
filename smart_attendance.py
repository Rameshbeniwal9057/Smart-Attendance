import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

def initialize():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")
    except cv2.error as e:
        print(f"Error initializing OpenCV modules: {e}")
        exit(1)
    return recognizer, face_classifier, eye_classifier

def create_directories():
    if not os.path.exists("Images"):
        os.makedirs("Images")
        print("Created folder: Images")
    if not os.path.exists("yml"):
        os.makedirs("yml")
        print("Created folder: yml")

def generate_dataset(id, name, face_classifier, eye_classifier):
    def face_and_eyes_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_classifier.detectMultiScale(roi_gray)
            if len(eyes) == 2:
                face_for_display = cropped_face.copy()
                return cropped_face, face_for_display

        return None, None

    cap = cv2.VideoCapture(0)
    print("Collecting 50 samples...")
    img_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                cropped_face, face_for_display = face_and_eyes_cropped(frame)
                if cropped_face is not None:
                    img_id += 1
                    face = cv2.resize(cropped_face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = f"Images/user.{id}.{name}.{img_id}.jpg"
                    cv2.imwrite(file_name_path, face)
                    cv2.imshow("Face with Eyes", face_for_display)
                    print(f"Image sample {img_id}")

                if cv2.waitKey(1) == 13 or img_id == 50:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Collecting samples completed...")

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNP = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNP)
        ids.append(id)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("yml/classifier.xml")
    print("Training completed and model saved as 'yml/classifier.xml'.")


def take_attendance(recognizer, face_classifier, eye_classifier):
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%d-%m-%Y")

    df = pd.read_csv("attendance.csv")
    recognizer.read("yml/classifier.xml")
    print("Model loaded successfully.")


    if formatted_date not in df.columns:
        df[formatted_date] = [""] * len(df)

    cam = cv2.VideoCapture(0)
    confidence_threshold = 50  # Confidence level threshold

    try:
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_classifier.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    if conf <= confidence_threshold:  # Check confidence
                        if not df[df.Id == id].empty:
                            name = df['Name'][df.Id == id].values[0]
                            cv2.putText(img, f"{name}", (x, y-9), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle
                            df.loc[df.Id == id, formatted_date] = 'P'
                            print(f"Attendance marked for {name} with confidence {100 - conf:.2f}%")
                        else:
                            print(f"ID {id} not found in the attendance list.")
                    else:
                        print(f"Face detected but confidence too low: {100 - conf:.2f}%")
                        cv2.putText(img, "Unknown", (x, y-9), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle for low confidence

            cv2.imshow('face', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        df.to_csv("attendance.csv", index=False)

def main():
    recognizer, face_classifier, eye_classifier = initialize()
    create_directories()

    while True:
        user_input = input('''************************* Smart Attendance **************************
                       
1. New User     
2. Attendance  
3. Exit 
                       
Press the Number : ''')
        print()

        if user_input == "1":
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%d-%m-%Y")
            initial_headers = ['Id', 'Name']
            name = input("Enter the Full Name: ")
            id = input("Enter the Id No: ")
            new_data = {'Id': id, 'Name': name, formatted_date: 'A'}

            file_path = 'attendance.csv'

            file_exists = os.path.isfile(file_path)
            rows = []
            if file_exists:
                with open(file_path, mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    existing_columns = reader.fieldnames

                    if formatted_date not in existing_columns:
                        existing_columns.append(formatted_date)
                    column_headers = existing_columns

                    for row in reader:
                        rows.append(row)
            else:
                column_headers = initial_headers + [formatted_date]

            id_exists = False

            for row in rows:
                if row['Id'] == new_data['Id']:
                    id_exists = True
                    row[formatted_date] = 'A'
                    break

            if id_exists:
                print(f"Data for ID [{id}] already exists and has been updated with today's attendance.")
                continue  # Skip to the next iteration of the loop

            rows.append(new_data)

            with open(file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=column_headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            print(f"New entry for ID [{id}] added to {file_path}.")

            generate_dataset(id, name, face_classifier, eye_classifier)
            train_classifier("Images")

        elif user_input == "2":
            take_attendance(recognizer, face_classifier, eye_classifier)

        elif user_input == "3":
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()

