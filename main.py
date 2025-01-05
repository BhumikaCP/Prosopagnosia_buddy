import os
import cv2
import numpy as np
import pyttsx3 #For Text to speech
from datetime import datetime

#Text to speech function
def speak_name(name, already_announced):
    """Uses pyttsx3 for Text-To-Speech to announce the recognized person's name once."""
    if name in already_announced:
        return already_announced
    
    engine=pyttsx3.init()
    engine.setProperty("rate", 150) #Speed of speech
    engine.setProperty("volume", 0.9) #Volume level (0-1)

    if name!="Unknown":
        engine.say(f"This is {name}.")
        already_announced.add(name) #Mark as announced
    else:
        engine.say("I'm sorry, do I know you?")

    engine.runAndWait()
    return already_announced


def preprocess_image(image_path):
    """Preprocess the image by detcting the face, cropping and resizing it."""
    face_cascade_path=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    face_cascade=cv2.CascadeClassifier(face_cascade_path)

    image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Image not found at path:{image_path}")
    
    faces=face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces)==0:
        raise ValueError("No faces detected in the image.")
    
    #Use first detected face
    x,y,w,h=faces[0]
    face_roi=image[y:y+h,x:x+w]
    face_resized=cv2.resize(face_roi, (100,100)) # Resize to match training dimensions
    return face_resized

def train_recognizer(training_dir):
    """Train the face recognizer with labeled images."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    #Training data
    training_data=[]
    labels=[]
    label_map={}

    if not os.path.exists(training_dir):
        print(f"Training directory '{training_dir}' not found.")
        return None, None, None
    
    label_id=0
    for label in os.listdir(training_dir):
        label_path=os.path.join(training_dir, label)
        if not os.path.isdir(label_path):
            continue

        label_map[label_id]=label
        for img_file in os.listdir(label_path):
            img_path=os.path.join(label_path, img_file)
            img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            training_data.append(img)
            labels.append(label_id)

        label_id+=1

    recognizer.train(training_data, np.array(labels))
    print("Training Complete!")
    return recognizer, label_map, face_cascade


def recognize_faces_live(recognizer, label_map, face_cascade):
    """Recognizes faces in real-time using webcam video and announces their name."""
    cap = cv2.VideoCapture(0)
    already_announced = set()  # Set to track announced names
    no_face_count = 0  # Counter to reset announcements when no faces are detected

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        #Convert to grayscale for face detection and recognition
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Detect faces in the frame
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5)
        if len(faces)==0:
            no_face_count+=1
            if no_face_count>30:
                already_announced.clear()
        else:
            no_face_count=0

        for (x,y,w,h) in faces:
            #Extract the face region of interest (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)
            
            name = label_map.get(label, "Unknown")
            confidence_text = f"{name} (Confidence: {confidence:.2f})"

            # Display on screen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Speak name
            already_announced = speak_name(name, already_announced)

        cv2.imshow("Recognition", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    training_dir = "training_images"  # Training directory path

    # Train the recognizer
    recognizer, label_map, face_cascade = train_recognizer(training_dir)
    if recognizer is None or label_map is None or face_cascade is None:
        print("Failed to initialize face recognition system. Exiting...")
    else:
        print("Training completed successfully!")
        # Start real-time face recognition
        recognize_faces_live(recognizer, label_map, face_cascade)

