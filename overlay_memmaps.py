import numpy as np
import cv2
from image_utils import get_num_frames



def overlay_videos(background_video_memmap,
                   foreground_video_memmap,
                   output_video_memmap,
                   height,
                   width,
                   channels=3):
    """
    Take the frames from the `foreground_video_memmap`, which are masked, and overlay
    them on top of the frames in the `background_video_memmap`. Both videos should be
    encoded with the same `height`, `width`, and `channels`.
    """
    # Determine the number of frames for each video
    background_num_frames = get_num_frames(background_video_memmap, height, width, channels)
    foreground_num_frames = get_num_frames(foreground_video_memmap, height, width, channels)

    # Set the number of frames for the output video
    num_frames = max(background_num_frames, foreground_num_frames)

    # Load the memmap arrays for background and foreground videos with the correct number of frames
    background_frames = np.memmap(background_video_memmap,
                                  dtype='uint8',
                                  mode='r',
                                  shape=(background_num_frames, height, width, channels))
    foreground_frames = np.memmap(foreground_video_memmap,
                                  dtype='uint8',
                                  mode='r',
                                  shape=(foreground_num_frames, height, width, channels))

    # Create memmap array for the output video
    output_frames = np.memmap(output_video_memmap,
                              dtype='uint8',
                              mode='w+',
                              shape=(num_frames, height, width, channels))

    for i in range(num_frames):
        if i < background_num_frames and i < foreground_num_frames:
            # Get the current frames from background and foreground videos
            background_frame = background_frames[i]
            foreground_frame = foreground_frames[i]

            # Create a binary mask from the foreground by checking where there are non-black pixels
            foreground_mask = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)
            _, foreground_mask = cv2.threshold(foreground_mask, 1, 255, cv2.THRESH_BINARY)

            # Invert the mask to create a mask for the background
            background_mask = cv2.bitwise_not(foreground_mask)

            # Apply the masks
            background_frame_masked = cv2.bitwise_and(background_frame,
                                                      background_frame,
                                                      mask=background_mask)
            foreground_frame_masked = cv2.bitwise_and(foreground_frame,
                                                      foreground_frame,
                                                      mask=foreground_mask)

            # Add the foreground frame on top of the masked background frame
            combined_frame = cv2.add(background_frame_masked, foreground_frame_masked)

        elif i < background_num_frames:
            # Only background frame is available
            combined_frame = background_frames[i]
        
        elif i < foreground_num_frames:
            # Only foreground frame is available
            combined_frame = foreground_frames[i]
        
        # Save the combined frame to the output memmap array
        output_frames[i] = combined_frame

    # Flush the output memmap to disk
    output_frames.flush()



if __name__ == "__main__":
    from image_utils import display_memmap_frames

    # Overlay two video files. NOTE: they may need to be generated.
    overlay_videos(background_video_memmap="output_video_1.dat",
                   foreground_video_memmap="output_video_2.dat",
                   output_video_memmap="overlayed_video_a.dat",
                   height=1440,
                   width=2560)

    # Display the result.
    display_memmap_frames(memmap_file="overlayed_video_a.dat",
                          height=1440,
                          width=2560)
