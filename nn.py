import cv2
import numpy as np

def save_video_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame as a .npy file
        frame_path = f"{output_dir}/frame_{frame_count:04d}.npy"
        np.save(frame_path, frame)
        frame_count += 1
    
    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}")

# Example usage
save_video_frames('video_c.avi', 'output_frames')
