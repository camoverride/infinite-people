"""
Produces masks by skipping every other frame and applying the mask from one frame to the next.
This means masks lag slightly behind movements, but this is mostly imperceptible.

This takes ~2:30 mins to process 10 secs of video on my MacBook Pro. It will probably be faster
on a GPU.

TODO: test on GPU, test YOLO model, test fast yolo bounding boxes instead of segmentations.
"""
import cv2
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights


def create_masks(path_to_video_file, output_frame_memmaps, output_frame_mask_memmaps):
    """
    Create maskes video frames saved as memmaps for fast I/O operations.

    TODO: `output_frame_mask_memmaps` are not used and can be removed.
    """
    # Load the DeepLabV3 model pre-trained on COCO dataset
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights).eval()

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
    # print(f"Frame count: {frame_count}")
    # print(f"Frame width: {frame_width}")
    # print(f"Frame height: {frame_height}")

    # Create memmap arrays for output
    masked_frames = np.memmap(output_frame_memmaps, dtype='uint8', mode='w+', shape=(frame_count, frame_height, frame_width, 3))
    masks = np.memmap(output_frame_mask_memmaps, dtype='uint8', mode='w+', shape=(frame_count, frame_height, frame_width))

    frame_index = 0

    # Use tqdm for progress bar
    with tqdm(total=frame_count) as pbar:
        while frame_index < frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (for the model)
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
            masked_frames[frame_index] = masked_frame  # Save in BGR format as OpenCV uses BGR
            masks[frame_index] = person_mask * 255  # Convert to 0-255 range

            pbar.update(1)  # Update the progress bar for the first frame

            # Apply the same mask and masked frame to the next frame
            frame_index += 1
            if frame_index < frame_count:
                ret, frame = cap.read()
                if not ret:
                    break
                masked_frames[frame_index] = masked_frame
                masks[frame_index] = person_mask * 255

                pbar.update(1)  # Update the progress bar for the second frame

            frame_index += 1

    # Close the video file
    cap.release()

    # Flush memmaps to disk
    masked_frames.flush()
    masks.flush()

    print(f"Masked video frames saved to {output_frame_memmaps}")
    print(f"Masks saved to {output_frame_mask_memmaps}")



if __name__ == "__main__":
    # Example usage
    pass

    # Get the memmaps for the first video
    # create_masks(path_to_video_file="new_video.mp4",
    #             output_frame_memmaps="new_video.dat",
    #             output_frame_mask_memmaps="new_video_mask.dat")

    # display_memmap_frames(memmap_file='output_video_1.dat',
    #                     num_frames=322,
    #                     height=1440,
    #                     width=2560)

    # # Get the memmaps for the second video
    # create_masks(path_to_video_file='output_video_2.mp4',
    #             output_frame_memmaps='output_video_2.dat',
    #             output_frame_mask_memmaps='output_video_2_mask.dat')

    # display_memmap_frames(memmap_file='output_video_2.dat',
    #                     num_frames=321,
    #                     height=1440,
    #                     width=2560)
