# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.apis import DetInferencer
import cv2
import os

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cpu')

def main():
    video_path = 'golden_2.mp4'
    output_video_path = 'batch_out/vid_vis.mp4'
    output_dir = 'batch_out/'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        result = inferencer(frame, show=False, return_vis=True)

        # Process predictions
        for pred in result['predictions']:
            labels = pred['labels']  # List of labels
            bboxes = pred['bboxes']  # List of bounding boxes
            scores = pred['scores']  # List of confidence scores
            for i in range(len(labels)):
                if labels[i] in [2, 7]:  # Labels for cars and trucks
                    # Crop the vehicle using the corresponding bounding box
                    x1, y1, x2, y2 = map(int, bboxes[i])  # Convert to integers
                    cropped_vehicle = frame[y1:y2, x1:x2]  # Cropping the vehicle
                    
                    # Save cropped image only if the score is greater than 0.3
                    if scores[i] > 0.3:  # Confidence threshold
                        unique_id = len(os.listdir(output_dir)) + 1  # Count existing files
                        cv2.imwrite(os.path.join(output_dir, f'cropped_{labels[i]}_{unique_id}.png'), cropped_vehicle)

        # Write the visualized frame to the output video
        out.write(result['visualization'][0])  # Accessing the visualization

    cap.release()
    out.release()

if __name__ == '__main__':
    main()

# Command to run the script
# python3 car_truck_crop.py golden_2.mp4 --out batch_out/vid_vis.mp4 --output-dir batch_out/