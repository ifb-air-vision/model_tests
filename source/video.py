import cv2
from ultralytics import YOLO
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and move it to the device
model = YOLO("model/yolov10n.pt").to(device)

# Open the video file
video_path = 'video.mp4'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was read correctly
    if not ret:
        break

    # Convert the frame to the appropriate format (if necessary)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Process the results
    annotated_frame = results[0].plot()  # Use plot() to draw annotations

    # Convert back to BGR for display with OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the annotated frame
    cv2.imshow('YOLOv10', annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
