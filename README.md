# YOLO Object Detection Video Annotation

This repository contains code for annotating videos using the YOLOv10 object detection model. The application processes each frame of a video, detects objects, annotates them, and saves both the annotated video and a JSON file with the detection results.

## Features

- **Object Detection:** Uses YOLOv10 to detect objects in each frame of a video.
- **Video Processing:** Reads a video file, annotates each frame with bounding boxes and labels, and writes the results to a new video file.
- **Results Export:** Saves detection results in a JSON file, including object classes, confidence scores, and bounding boxes.

## Requirements

- Python 3.7 or later
- `cv2` (OpenCV) - for video processing and annotation
- `ultralytics` - for YOLO object detection
- `torch` - for PyTorch backend

You can install the required Python packages using pip:

```bash
pip install opencv-python ultralytics torch
```

## Usage

1. **Prepare the Model:**
   - Download the YOLOv10 model weights (`best.pt`) and place it in the project directory.

2. **Prepare the Video:**
   - Place your video file in the `test_objects` directory or update the `video_path` variable in the code to point to your video file.

3. **Run the Script:**

   Save the provided code in a file named `videoResNativa.py` and run:

   ```bash
   python videoResNativa.py
   ```

4. **Outputs:**
   - The annotated video will be saved as `annotated_output.mp4`.
   - Detection results for each frame will be saved in `predictions.json`.

## Code Overview

1. **Device Setup:**
   - Checks if CUDA is available and sets the device to GPU or CPU accordingly.

2. **Model Loading:**
   - Loads the YOLOv10 model (`best.pt`) and moves it to the selected device.

3. **Video Processing:**
   - Opens the video file, retrieves video properties, and sets up a `VideoWriter` to save the annotated video.
   - Processes each frame, performs object detection, annotates the frame, and saves it to the output video.

4. **Results Handling:**
   - Collects detection results for each frame and saves them in a JSON file.

5. **Resource Management:**
   - Releases video capture and writer resources, and closes any OpenCV windows.

## Configuration

- **Model Path:** Update the `YOLO` constructor argument with the path to your model weights if different from `best.pt`.
- **Video Path:** Update the `video_path` variable with the path to your input video file.
- **Output Paths:** Modify `output_path` and JSON file paths as needed.

## Troubleshooting

- Ensure that the video file path is correct and the file is accessible.
- Make sure all required packages are installed and compatible with your Python version.
- Check the console output for any errors or warnings during execution.
