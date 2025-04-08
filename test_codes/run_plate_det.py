from fast_alpr import ALPR
import cv2
import os
import numpy as np
from mmdet.apis import DetInferencer



def detect_plates(frame):
    # Detect plates in the given frame
    alpr_results = alpr.predict(frame)
    return alpr_results

def crop_and_detect_plates(frame, bboxes):
    # Crop the vehicle boxes and detect plates in each box
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cropped_vehicle = frame[y1:y2, x1:x2]
        plate_results = detect_plates(cropped_vehicle)
        # Draw plate boxes on the cropped vehicle
        for plate in plate_results:
            # Assuming plate has 'bbox' key for bounding box
            plate_bbox = plate['bbox']
            # Adjust the plate coordinates based on the vehicle's bounding box
            cv2.rectangle(frame, (x1 + plate_bbox[0], y1 + plate_bbox[1]), (x1 + plate_bbox[2], y1 + plate_bbox[3]), (0, 255, 0), 2)
        # Display or process the cropped vehicle with detected plates as needed

def runImage(image_path):
    b_cropMode = True  # Set to True for crop mode, False for non-crop mode  # Specify your image path
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return

    if b_cropMode:
        #initialize the car truck detecter
        inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cpu')
        MMDETresult = inferencer(frame, show=False, return_vis=False)
        for pred in MMDETresult['predictions']:
            labels = pred['labels']  # List of labels
            bboxes = pred['bboxes']  # List of bounding boxes
            scores = pred['scores']  # List of confidence scores
            for i in range(len(labels)):
                if labels[i] in [2, 7] and scores[i] > 0.5:  # Labels for cars and trucks
                    # Crop the vehicle using the corresponding bounding box
                    DETx1, DETy1, DETx2, DETy2 = map(int, bboxes[i])  # Convert to integers
                    cropped_vehicle_frame = frame[DETy1:DETy2, DETx1:DETx2]  # Cropping the vehicle
                    plate_results = detect_plates(cropped_vehicle_frame)
                    for plate in plate_results:
                        # Extract coordinates from plate_bbox
                        x1 = plate.detection.bounding_box.x1
                        y1 = plate.detection.bounding_box.y1
                        x2 = plate.detection.bounding_box.x2
                        y2 = plate.detection.bounding_box.y2
                        cv2.rectangle(frame, (DETx1 + x1, DETy1 + y1), (DETx1 + x2, DETy1 + y2), (0, 255, 0), 2)

        # Detect cars/trucks first (this part should be implemented)
        # Assuming we have a function `detect_cars_trucks(frame)` that returns bounding boxes
        # bboxes = detect_cars_trucks(frame)  # Placeholder for car/truck detection
        # crop_and_detect_plates(frame, bboxes)
        pass
    else:
        # Directly detect plates from the image
        plate_results = detect_plates(frame)
        for plate in plate_results:
            # Extract coordinates from plate_bbox
            x1 = plate.detection.bounding_box.x1
            y1 = plate.detection.bounding_box.y1
            x2 = plate.detection.bounding_box.x2
            y2 = plate.detection.bounding_box.y2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detection Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize the ALPR for plate detection
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
    )
    image_path = 'Screenshot 2025-04-04 at 01.43.20.png'
    runImage(image_path)