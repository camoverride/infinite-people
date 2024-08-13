import cv2
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

def create_composite_video(video_a_path, video_b_path, output_path):
    # Load YOLOv5 nano model
    model = DetectMultiBackend('yolov5n.pt', device='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    
    # Open video files
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)
    
    # Check if videos are opened successfully
    if not cap_a.isOpened() or not cap_b.isOpened():
        print("Error: Could not open one of the input video files.")
        return
    
    # Get video properties from the first video
    width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_a.get(cv2.CAP_PROP_FPS)
    
    # Check if FPS is valid
    if fps <= 0:
        fps = 30  # default FPS
    
    # Video writers for mask videos
    out_a_mask = cv2.VideoWriter('video_a_mask.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
    out_b_mask = cv2.VideoWriter('video_b_mask.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
    
    # Video writer for the final output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    if not out.isOpened() or not out_a_mask.isOpened() or not out_b_mask.isOpened():
        print("Error: Could not open one of the output video files for writing.")
        return
    
    frame_count = 0
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a and not ret_b:
            break
        
        if ret_a:
            mask_a = create_mask_from_yolo_results(frame_a, model, width, height)
            out_a_mask.write(mask_a)
        else:
            mask_a = np.zeros((height, width), dtype=np.uint8)
        
        if ret_b:
            mask_b = create_mask_from_yolo_results(frame_b, model, width, height)
            out_b_mask.write(mask_b)
        else:
            mask_b = np.zeros((height, width), dtype=np.uint8)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_a, mask_b)
        
        # Extract motion areas from both videos
        motion_a = cv2.bitwise_and(frame_a, frame_a, mask=mask_a) if ret_a else np.zeros_like(frame_b)
        motion_b = cv2.bitwise_and(frame_b, frame_b, mask=mask_b) if ret_b else np.zeros_like(frame_a)
        
        # Combine the motion frames
        combined_motion = cv2.addWeighted(motion_a, 0.5, motion_b, 0.5, 0)
        
        # Convert the grayscale background to BGR before adding it to the combined motion
        background_bgr = np.zeros_like(combined_motion)  # Assuming background is black
        final_frame = cv2.add(background_bgr, combined_motion)
        
        # Write the final frame to the output video
        out.write(final_frame)
        frame_count += 1
        print(f"Processed frame {frame_count}")
    
    print("Mask videos and final video created successfully.")
    
    # Release resources
    cap_a.release()
    cap_b.release()
    out.release()
    out_a_mask.release()
    out_b_mask.release()
    cv2.destroyAllWindows()

def create_mask_from_yolo_results(frame, model, width, height):
    # Preprocess the frame for YOLOv5
    img = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(model.device)
    
    # Run inference
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)
    
    # Initialize an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Ensure that pred is not empty and contains valid detections
    if pred is not None and len(pred) > 0 and pred[0] is not None and len(pred[0]) > 0:
        for det in pred[0]:  # Loop through detections
            if det is not None and len(det) == 6:  # Each detection should contain 6 elements: [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    mask[y1:y2, x1:x2] = 255
    
    return mask



# Example usage
create_composite_video('video_a.avi', 'video_b.avi', 'video_c.avi')
