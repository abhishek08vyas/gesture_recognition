import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

def initialize_label_map():
    """Initialize the label map for gestures detected by the YOLO model"""
    return {
        0: 'grabbing', 1: 'grip', 2: 'holy', 3: 'point', 4: 'call', 5: 'three3', 
        6: 'timeout', 7: 'xsign', 8: 'hand_heart', 9: 'hand_heart2', 10: 'little_finger', 
        11: 'middle_finger', 12: 'take_picture', 13: 'dislike', 14: 'fist', 15: 'four', 
        16: 'like', 17: 'mute', 18: 'ok', 19: 'one', 20: 'palm', 21: 'peace', 
        22: 'peace_inverted', 23: 'rock', 24: 'stop', 25: 'stop_inverted', 26: 'three', 
        27: 'three2', 28: 'two_up', 29: 'two_up_inverted', 30: 'three_gun', 
        31: 'thumb_index', 32: 'thumb_index2', 33: 'no_gesture'
    }

def real_time_asl_recognition():
  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    model = YOLO('./YOLOv10n_gestures.pt').to(device)

    # Initialize label map
    label_map = initialize_label_map()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize FPS calculation
    prev_time = 0
    
    print("Starting real-time gesture recognition with YOLO. Press 'q' to quit.")
    print("Available signs:", ", ".join(label_map.values()))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Run the frame through the YOLO model
        results = model(frame)
        
        # Parse results (bounding boxes, labels, confidence scores)
        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                # Extract bounding box coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box
                confidence = box.conf[0].cpu().numpy()  # Confidence
                class_id = int(box.cls[0].cpu().numpy())  # Class ID
                
                # Only display results with high confidence (e.g., > 0.5)
                if confidence > 0.5:
                    predicted_gesture = label_map.get(class_id, "Unknown Gesture")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add prediction text to frame
                    text = f"Prediction: {predicted_gesture} ({confidence:.2f})"
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (550, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

try:
    print("Initializing real-time gesture recognition with YOLO...")
    real_time_asl_recognition()
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Ensure webcam is released even if an error occurs
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()
