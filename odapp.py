import cv2
import numpy as np
import streamlit as st

# Load YOLOv3 model and configuration
weight = 'yolov3.weights'
cfg = 'yolov3.cfg'
net = cv2.dnn.readNet(weight, cfg)

# Load class names from COCO dataset
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Streamlit app setup
st.title("Real-Time Object Detection with YOLOv3")

# Initialize webcam with smaller resolution (640x480) to speed up processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")

# Display the frame in Streamlit
FRAME_WINDOW = st.image([])

# Function to detect objects in a frame
def detect_objects(frame):
    height, width, channels = frame.shape

    # Object detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process YOLO detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to avoid duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green for objects
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return frame

# Button to start/stop object detection
if st.button('Start/Stop Object Detection'):
    run_detection = not st.session_state.get('run_detection', False)
    st.session_state.run_detection = run_detection

# Main loop for object detection
if st.session_state.get('run_detection', False):
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        # Detect objects and update frame
        frame = detect_objects(frame)

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        # Check if the user has stopped detection
        if not st.session_state.run_detection:
            break

# Release the webcam when the app is stopped
cap.release()