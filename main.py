import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv

# Initialize models and tracking
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(r'C:\Users\moh19\PycharmProjects\pythonProject9\license_plate_detector (3).pt')
mot_tracker = Sort()

# Load video
cap = cv2.VideoCapture(r"C:\Users\moh19\Downloads\nr.mp4")

# Define vehicle classes based on COCO dataset
vehicles = [2, 3, 5, 7]

# Initialize results dictionary
results = {}

frame_nmr = -1
ret = True

# Loop through frames
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        # Initialize results for the current frame
        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [
            [x1, y1, x2, y2, score] for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
            if int(class_id) in vehicles
        ]

        # Track vehicles using SORT
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Convert to grayscale and apply binary threshold
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Store results if license plate text is detected
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

# Write results to CSV
write_csv(results, './test.csv')

# Release video capture
cap.release()
