import cv2
import torch
import numpy as np
from torchvision import models
from torchvision.transforms import functional as F

def create_composite_video(video_a_path, video_b_path, background_img_path, output_path):
    # Load DeepLabV3 model (pretrained on COCO dataset)
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    
    # Open video files
    cap_a = cv2.VideoCapture(video_a_path)
    cap_b = cv2.VideoCapture(video_b_path)
    
    # Load the background image
    background_img = cv2.imread(background_img_path)
    background_img = cv2.resize(background_img, (int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
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
            mask_a = create_mask_from_deeplabv3(frame_a, model, width, height)
            out_a_mask.write(mask_a)
        else:
            mask_a = np.zeros((height, width), dtype=np.uint8)
        
        if ret_b:
            mask_b = create_mask_from_deeplabv3(frame_b, model, width, height)
            out_b_mask.write(mask_b)
        else:
            mask_b = np.zeros((height, width), dtype=np.uint8)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_a, mask_b)
        intersection_mask = cv2.bitwise_and(mask_a, mask_b)
        
        # Extract motion areas from both videos
        motion_a = cv2.bitwise_and(frame_a, frame_a, mask=mask_a) if ret_a else np.zeros_like(frame_b)
        motion_b = cv2.bitwise_and(frame_b, frame_b, mask=mask_b) if ret_b else np.zeros_like(frame_a)
        
        # Prepare the final frame
        final_frame = background_img.copy()
        
        # Add motion from Video A and Video B where the masks are
        final_frame = np.where(mask_a[:, :, np.newaxis] == 255, motion_a, final_frame)
        final_frame = np.where(mask_b[:, :, np.newaxis] == 255, motion_b, final_frame)
        
        # Blend the overlapping regions
        blended_motion = cv2.addWeighted(motion_a, 0.5, motion_b, 0.5, 0)
        final_frame = np.where(intersection_mask[:, :, np.newaxis] == 255, blended_motion, final_frame)
        
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

def create_mask_from_deeplabv3(frame, model, width, height):
    # Convert the frame to a PIL image and apply transformations
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = F.to_tensor(input_image).unsqueeze(0)

    # Run the model on the input frame
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # Create a binary mask for 'person' class (class 15 in COCO dataset)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[output_predictions == 15] = 255
    
    return mask

# Example usage
create_composite_video('video_a.avi', 'video_b.avi', 'background.jpg', 'video_c.avi')
