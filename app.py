import os
import cv2
import numpy as np
import joblib
import streamlit as st
import pyttsx3
from threading import Thread
from PIL import Image
from deepface import DeepFace
from ultralytics import YOLO
import time

# Setting up the Streamlit page
st.set_page_config(
    page_title="Object and Face Recognition",
    layout="wide"
)

# Simple thread function for text-to-speech to avoid UI blocking
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()

def speak_in_thread(text):
    thread = Thread(target=speak_text, args=(text,))
    thread.daemon = True
    thread.start()

def main():
    st.title("Object and Face Recognition System")
    
    # Interface elements
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        face_recognition_threshold = st.slider("Face Recognition Threshold", 1.0, 10.0, 5.0, 0.5)
        enable_voice = st.checkbox("Enable Voice", value=True)
        
        st.subheader("Controls")
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")
        
        st.subheader("Detected Objects")
        detected_objects_placeholder = st.empty()
        
        st.subheader("Recognized Faces")
        recognized_faces_placeholder = st.empty()
    
    with col1:
        st.subheader("Camera Feed")
        # Create a placeholder for the webcam feed
        video_placeholder = st.empty()
    
    # Load YOLOv8 model
    @st.cache_resource
    def load_model():
        return YOLO("yolov5s.pt")
    
    model = load_model()
    
    # File path for storing embeddings
    embedding_file = "face_embeddings.joblib"
    
    # Load known face embeddings
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            known_embeddings = joblib.load(f)
        st.sidebar.success("Loaded existing face embeddings.")
    else:
        st.sidebar.warning("No face embeddings found! Run face embedding generation first.")
        known_embeddings = {}
    
    # Variables to track detected objects and faces
    detected_objects_set = set()
    recognized_faces_set = set()
    last_spoken = ""
    last_face_spoken = None
    last_spoken_time = time.time() - 10  # Initialize with a time 10 seconds ago
    
    # Only run the webcam if the start button is pressed
    if start_button:
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened(0):
            st.error("Could not open webcam. Please check your camera connection.")
            return
        
        frame_width = int(cap.get(3))  # Get frame width
        
        # Run until the stop button is pressed
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break
            
            # Run YOLOv8 object detection
            results = model(frame)
            
            # Clear the sets for this frame
            current_objects = set()
            current_faces = set()
            
            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    label = result.names[int(box.cls[0])]
                    
                    if confidence > confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Determine object position
                        center_x = (x1 + x2) // 2  
                        position = "center"
                        if center_x < frame_width // 3:
                            position = "left"
                        elif center_x > 2 * frame_width // 3:
                            position = "right"
                        
                        if label == "person":
                            face_crop = frame[y1:y2, x1:x2]  # Crop face area
                            
                            try:
                                # Use DeepFace to detect a real face first
                                detected_faces = DeepFace.extract_faces(face_crop, enforce_detection=False)
                                
                                if not detected_faces:
                                    recognized_person = "Unknown Person"
                                else:
                                    # Extract embedding only if a real face is found
                                    face_embedding = DeepFace.represent(face_crop, enforce_detection=True)[0]["embedding"]
                                    face_embedding = np.array(face_embedding)
                                    
                                    # Default to "Unknown Person"
                                    recognized_person = "Unknown Person"
                                    min_distance = float("inf")
                                    
                                    # Compare with stored embeddings
                                    for name, known_embedding in known_embeddings.items():
                                        distance = np.linalg.norm(known_embedding - face_embedding)
                                        
                                        if distance < min_distance:
                                            min_distance = distance
                                            recognized_person = name
                                    
                                    # If no match is found within threshold, mark as "Unknown Person"
                                    if min_distance >= face_recognition_threshold:
                                        recognized_person = "Unknown Person"
                                
                                # Add to current faces with position information
                                face_info = f"{recognized_person} on the {position}"
                                current_faces.add(face_info)
                                
                                # Directly pronounce the detected face with position
                                current_time = time.time()
                                if enable_voice and face_info != last_face_spoken and current_time - last_spoken_time > 3:
                                    speak_in_thread(f"{recognized_person} on the {position}")
                                    last_face_spoken = face_info
                                    last_spoken_time = current_time
                                
                                # Draw bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, face_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            except Exception as e:
                                # If face processing fails, mark as Unknown
                                recognized_person = "Unknown Person"
                                face_info = f"{recognized_person} on the {position}"
                                current_faces.add(face_info)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, face_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        else:
                            # For non-person objects
                            object_info = f"{label} on the {position}"
                            current_objects.add(object_info)
                            
                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, object_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Directly pronounce the detected object with position
                            current_time = time.time()
                            if enable_voice and object_info != last_spoken and current_time - last_spoken_time > 3:
                                speak_in_thread(f"{label} on the {position}")
                                last_spoken = object_info
                                last_spoken_time = current_time
            
            # Update the sets of detected objects and faces
            detected_objects_set = current_objects
            recognized_faces_set = current_faces
            
            # Update the UI with current detections
            detected_objects_placeholder.write("\n".join(detected_objects_set) if detected_objects_set else "None")
            recognized_faces_placeholder.write("\n".join(recognized_faces_set) if recognized_faces_set else "None")
            
            # Convert the frame to RGB (from BGR) for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Using use_container_width instead of deprecated use_column_width
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Release the webcam when stopping
        cap.release()
        video_placeholder.empty()
        st.success("Camera stopped")

if __name__ == "__main__":
    main()