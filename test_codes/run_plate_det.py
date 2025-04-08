from fast_alpr import ALPR
import cv2
import os
import numpy as np
from mmdet.apis import DetInferencer


def runImage(frame):
    b_cropMode = False  # Set to True for crop mode, False for non-crop mode  # Specify your image path
    
    if b_cropMode:
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
                    plate_results = alpr.predict(cropped_vehicle_frame)
                    for plate in plate_results:
                        # Extract coordinates from plate_bbox
                        x1 = plate.detection.bounding_box.x1
                        y1 = plate.detection.bounding_box.y1
                        x2 = plate.detection.bounding_box.x2
                        y2 = plate.detection.bounding_box.y2
                        cv2.rectangle(frame, (DETx1 + x1, DETy1 + y1), (DETx1 + x2, DETy1 + y2), (0, 255, 0), 2)
    else:
        # Directly detect plates from the image
        plate_results = alpr.predict(frame)
        for plate in plate_results:
            # Extract coordinates from plate_bbox
            x1 = plate.detection.bounding_box.x1
            y1 = plate.detection.bounding_box.y1
            x2 = plate.detection.bounding_box.x2
            y2 = plate.detection.bounding_box.y2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    # cv2.imshow("Detection Result", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return frame

def runVideo(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the width and height of the video frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use runImage to process the frame
        processed_frame = runImage(frame)  # Call runImage with the current frame

        # Write the processed frame to the output video
        out.write(processed_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize the ALPR for plate detection
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
    )
    # vehicle detector
    inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cpu')
    # image_path = 'Screenshot 2025-04-04 at 01.43.20.png'
    # frame = cv2.imread(image_path)
    # runImage(frame)
    video_path = 'golden_4.mp4'  # Specify your video path
    output_path = 'golden_4_noCrop.mp4'  # Specify the output video path
    runVideo(video_path, output_path)