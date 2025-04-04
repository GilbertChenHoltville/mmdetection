from mmdet.apis import DetInferencer
import cv2

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco', device = 'cpu')

# Perform inference
result = inferencer('./sole_car.png', show=False, 
           out_dir='./test_outputs/', 
            
           no_save_vis=False,
            # return_datasamples=True, 
            no_save_pred=False,
            return_vis=True)
# print(result['predictions'])
