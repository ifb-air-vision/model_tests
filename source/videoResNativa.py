import cv2
from ultralytics import YOLO
import torch
import json

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and move it to the device
model = YOLO("yolov10n.pt").to(device)

# Open the video file
video_path = 'test_objects/people.mov'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the annotated video
output_path = 'annotated_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Create a list to hold all frame predictions
all_predictions = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was read correctly
    if not ret:
        break

    # Convert the frame to RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Process results
    frame_predictions = []
    for result in results:
        # Extract bounding boxes, classes, and probabilities
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        classes = result.names  # Class names

        for i, box in enumerate(boxes):
            bbox = box.tolist()  # Convert to list
            class_id = int(result.boxes.cls[i].item()) if result.boxes.cls is not None else -1  # Handle None
            class_name = classes[class_id] if class_id >= 0 and class_id < len(classes) else "Unknown"
            confidence = result.boxes.conf[i].item() if result.boxes.conf is not None else 0.0  # Handle None

            # Prepare prediction entry
            prediction = {
                'class': class_name,
                'confidence': float(confidence),
                'bbox': bbox
            }
            frame_predictions.append(prediction)

    # Append predictions to the list for this frame
    all_predictions.append({
        'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),  # Frame number
        'predictions': frame_predictions
    })

    # Get the annotated frame
    annotated_frame = results[0].plot()  # Use plot() to get the annotated image

    # Convert back to BGR format for OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Resize the annotated frame to the original size if necessary
    if (annotated_frame.shape[1], annotated_frame.shape[0]) != (frame_width, frame_height):
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

    # Display the annotated frame
    cv2.imshow('YOLOv10', annotated_frame)

    # Save the annotated frame to the output video file
    out.write(annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save all predictions to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(all_predictions, f, indent=4)
