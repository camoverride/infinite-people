import cv2
import numpy as np

def create_composite_video(video_a_path, video_b_path, background_img_path, output_path):
    # Load the background image
    background_img = cv2.imread(background_img_path)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    
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
    
    # Resize background image to match video dimensions
    background_img = cv2.resize(background_img, (width, height))
    
    # Video writers for mask videos
    out_a_mask = cv2.VideoWriter('video_a_mask.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
    out_b_mask = cv2.VideoWriter('video_b_mask.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), isColor=False)
    
    # Video writer for the final output
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    if not out.isOpened() or not out_a_mask.isOpened() or not out_b_mask.isOpened():
        print("Error: Could not open one of the output video files for writing.")
        return
    
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        
        if not ret_a and not ret_b:
            break
        
        if ret_a:
            # Convert frame to grayscale
            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            # Subtract the background image
            diff_a = cv2.absdiff(gray_a, background_img)
            # Threshold the difference to get a binary mask
            _, mask_a = cv2.threshold(diff_a, 50, 255, cv2.THRESH_BINARY)
            # Refine the mask using morphological operations
            mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # Write the mask to the video
            out_a_mask.write(mask_a)
        else:
            mask_a = np.zeros((height, width), dtype=np.uint8)
        
        if ret_b:
            # Convert frame to grayscale
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            # Subtract the background image
            diff_b = cv2.absdiff(gray_b, background_img)
            # Threshold the difference to get a binary mask
            _, mask_b = cv2.threshold(diff_b, 50, 255, cv2.THRESH_BINARY)
            # Refine the mask using morphological operations
            mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # Write the mask to the video
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
        background_bgr = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)
        final_frame = cv2.add(background_bgr, combined_motion)
        
        # Write the final frame to the output video
        out.write(final_frame)
    
    print("Mask videos and final video created successfully.")
    
    # Release resources
    cap_a.release()
    cap_b.release()
    out.release()
    out_a_mask.release()
    out_b_mask.release()
    cv2.destroyAllWindows()

# Example usage
create_composite_video('video_a.avi', 'video_b.avi', 'background.jpg', 'video_c.avi')
