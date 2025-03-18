import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# 1. Load the trained model
model = load_model('emotion_detection_model.h5')
"""
Load a pre-trained Keras model for emotion detection from a .h5 file.
The model should have been trained on facial expression data to classify emotions.
"""

# 2. Emotion label dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
"""
A dictionary mapping numeric labels (from the model's output) to human-readable emotion names.
These emotions correspond to the classes that the model can predict.
"""

# 3. Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
"""
Load the Haar Cascade classifier from OpenCV for detecting faces in images.
The classifier is pre-trained and provided by OpenCV, which detects frontal faces.
"""

# 4. Initialize the GUI window
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("800x600")
"""
Create a Tkinter window titled "Emotion Detector" with a specified size of 800x600 pixels.
This window will serve as the main interface for the application.
"""

# 5. Create a label for displaying the video feed
video_label = Label(root)
video_label.pack()
"""
Create a Tkinter Label widget to display the video feed from the webcam.
The label will be updated with the frames captured from the webcam.
"""

# Global variables to control the webcam state
cap = None
running = False
"""
Initialize global variables:
- cap: To hold the video capture object for accessing the webcam.
- running: A boolean flag to indicate whether the webcam is currently running.
"""

# 6. Function to start the webcam and begin detecting emotions
def start_webcam():
    global cap, running
    cap = cv2.VideoCapture(0)  # Open the default webcam (0 is the index for the default camera)
    running = True  # Set the running flag to True
    detect_emotion()  # Start the emotion detection process

# 7. Function to stop the webcam
def stop_webcam():
    global cap, running
    running = False  # Set the running flag to False
    if cap is not None:
        cap.release()  # Release the webcam resource
    cv2.destroyAllWindows()  # Close any OpenCV windows

# 8. Function to detect faces and emotions, and update the GUI with the webcam feed
def detect_emotion():
    if not running:  # Check if the webcam is running
        return

    ret, frame = cap.read()  # Capture a frame from the webcam
    if ret:  # If frame is captured successfully
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]  # Region of Interest (ROI) for the detected face
            roi_resized = cv2.resize(roi_gray, (48, 48))  # Resize to 48x48 pixels for the model input
            roi_normalized = roi_resized / 255.0  # Normalize pixel values to [0, 1]
            roi_normalized = np.reshape(roi_normalized, (1, 48, 48, 1))  # Reshape for model input

            # Predict the emotion
            prediction = model.predict(roi_normalized)  # Get the model prediction
            max_index = int(np.argmax(prediction))  # Get the index of the highest probability
            predicted_emotion = emotion_dict[max_index]  # Retrieve the corresponding emotion label

            # Draw a rectangle around the face and put the emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle
            cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Label emotion

        # Convert the frame to RGB for Tkinter
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB format
        imgtk = ImageTk.PhotoImage(image=img)  # Create an ImageTk object

        # Update the GUI window with the video frame
        video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
        video_label.configure(image=imgtk)  # Update the label with the new image

    video_label.after(10, detect_emotion)  # Call this function again after 10 milliseconds

# 9. Add Start and Stop buttons to control the webcam
start_button = Button(root, text="Start Webcam", command=start_webcam)
start_button.pack(side="left", padx=10, pady=10)  # Pack the start button on the left

stop_button = Button(root, text="Stop Webcam", command=stop_webcam)
stop_button.pack(side="right", padx=10, pady=10)  # Pack the stop button on the right

# 10. Start the Tkinter main event loop
root.mainloop()
"""
Run the Tkinter event loop to keep the application running and responsive.
The loop will wait for user interactions and update the GUI as necessary.
"""