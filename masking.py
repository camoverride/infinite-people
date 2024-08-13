import cv2
import numpy as np
import os
from tqdm import tqdm
from torchvision import models
import torch
from torchvision import transforms
import time

def create_masks(path_to_video_file, output_frame_memmaps, output_frame_mask_memmaps):
    # Load the DeepLabV3 model pre-trained on COCO dataset
    model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).eval()

    # Define the transformation for the input image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open the video file
    cap = cv2.VideoCapture(path_to_video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print the debug information
    print(f"Frame count: {frame_count}")
    print(f"Frame width: {frame_width}")
    print(f"Frame height: {frame_height}")

    # Create memmap arrays for output
    masked_frames = np.memmap(output_frame_memmaps, dtype='uint8', mode='w+', shape=(frame_count, frame_height, frame_width, 3))
    masks = np.memmap(output_frame_mask_memmaps, dtype='uint8', mode='w+', shape=(frame_count, frame_height, frame_width))

    frame_index = 0

    # Process each frame
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the transformation and make the image batch-like
        input_tensor = preprocess(image_rgb).unsqueeze(0)

        # Get the segmentation mask
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Mask the person class (COCO class 15 corresponds to person)
        person_mask = (output_predictions == 15).astype(np.uint8)

        # Create masked frame by applying the mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=person_mask)

        # Save to memmap arrays
        masked_frames[frame_index] = masked_frame
        masks[frame_index] = person_mask * 255  # Convert to 0-255 range

        frame_index += 1

    # Close the video file
    cap.release()

    # Flush memmaps to disk
    masked_frames.flush()
    masks.flush()

    print(f"Masked video frames saved to {output_frame_memmaps}")
    print(f"Masks saved to {output_frame_mask_memmaps}")



def display_memmap_frames(memmap_file, num_frames, height, width, channels=3):
    # Load the memmap array with the correct shape
    frames = np.memmap(memmap_file, dtype='uint8', mode='r', shape=(num_frames, height, width, channels))

    for i in range(num_frames):
        # Read the frame from the memmap array
        frame = frames[i]

        # Manually swap channels from RGB to BGR if the image has 3 channels
        if channels == 3:
            frame = frame[:, :, [2, 1, 0]]  # Convert RGB to BGR

        # Display the frame
        cv2.imshow("Frame", frame)

        # Wait a short duration (0.01 seconds) before showing the next frame
        time.sleep(0.01)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            break

    # Close all windows
    cv2.destroyAllWindows()


display_memmap_frames('masked_frames_2.dat', num_frames=145, height=480, width=640)
# display_memmap_frames('masks_2.dat', num_frames=145, height=480, width=640, channels=1)

# Example usage
# create_masks('video_b.avi', 'masked_frames_2.dat', 'masks_2.dat')
