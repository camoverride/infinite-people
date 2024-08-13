import os
import numpy as np

# Set known parameters
memmap_path = 'masked_frames_2.dat'  # Replace with your memmap path
height = 480  # Replace with your known frame height
width = 640   # Replace with your known frame width
channels = 3  # Replace with the number of channels
dtype_size = 4  # For float32, dtype_size is 4 bytes

# Get the actual file size in bytes
actual_file_size = os.path.getsize(memmap_path)

# Calculate the total number of pixels per frame (height * width * channels)
pixels_per_frame = height * width * channels

# Calculate the number of frames in the memmap file
num_frames = actual_file_size // (pixels_per_frame * dtype_size)

# Load the memmap video with the calculated shape
memmap_video = np.memmap(memmap_path, dtype='float32', mode='r', shape=(num_frames, height, width, channels))

# Now you can proceed with the rest of your code

import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variable to keep track of memmap frames
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to match the memmap frame size
    frame = cv2.resize(frame, (width, height))

    # Retrieve the current frame from the memmap video
    memmap_frame = memmap_video[frame_index]

    # Ensure memmap_frame values are within [0, 1]
    memmap_frame = np.clip(memmap_frame, 0, 1)

    # Convert both frames to uint8
    memmap_frame = (memmap_frame * 255).astype(np.uint8)
    frame = frame.astype(np.uint8)

    # Overlay the memmap_frame onto the current frame
    combined_frame = cv2.addWeighted(frame, 0.7, memmap_frame, 0.3, 0)

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

