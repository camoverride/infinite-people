import cv2
import numpy as np
import os

def combine_memmap_with_video(memmap_path, num_frames, height, width, channels=3, alpha=0.3):
    # Load the memmap video
    memmap_video = np.memmap(memmap_path, dtype='uint8', mode='r', shape=(num_frames, height, width, channels))

    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)

    # Variable to keep track of memmap frames
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to match the memmap frame size (if necessary)
        frame = cv2.resize(frame, (width, height))

        # Retrieve the current frame from the memmap video
        memmap_frame = memmap_video[frame_index]

        # Ensure the memmap frame has the same dimensions as the incoming frame
        if memmap_frame.shape != frame.shape:
            memmap_frame = cv2.resize(memmap_frame, (width, height))

        # Combine the memmap_frame onto the current frame using cv2.addWeighted
        combined_frame = cv2.addWeighted(frame, 1 - alpha, memmap_frame, alpha, 0)

        # Display the combined frame
        cv2.imshow('Overlayed Video', combined_frame)

        # Update the frame index for the memmap video
        frame_index = (frame_index + 1) % num_frames

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
combine_memmap_with_video(
    memmap_path='masked_frames_2.dat', 
    num_frames=145, 
    height=480, 
    width=640,
    channels=3,
    alpha=0.3
)
