import numpy as np
import cv2
import time

def overlay_videos(background_video_memmap, foreground_video_memmap, output_video_memmap, num_frames, height, width, channels=3):
    # Load the memmap arrays for background and foreground videos
    background_frames = np.memmap(background_video_memmap, dtype='uint8', mode='r', shape=(num_frames, height, width, channels))
    foreground_frames = np.memmap(foreground_video_memmap, dtype='uint8', mode='r', shape=(num_frames, height, width, channels))

    # Create memmap array for the output video
    output_frames = np.memmap(output_video_memmap, dtype='uint8', mode='w+', shape=(num_frames, height, width, channels))

    for i in range(num_frames):
        # Get the current frames from background and foreground videos
        background_frame = background_frames[i]
        foreground_frame = foreground_frames[i]

        # Create a binary mask from the foreground by checking where there are non-black pixels
        foreground_mask = cv2.cvtColor(foreground_frame, cv2.COLOR_BGR2GRAY)
        _, foreground_mask = cv2.threshold(foreground_mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the mask to create a mask for the background
        background_mask = cv2.bitwise_not(foreground_mask)

        # Zero out the pixels in the background frame where the foreground mask is non-zero
        background_frame_masked = cv2.bitwise_and(background_frame, background_frame, mask=background_mask)

        # Add the foreground frame on top of the masked background frame
        combined_frame = cv2.add(background_frame_masked, foreground_frame)

        # Save the combined frame to the output memmap array
        output_frames[i] = combined_frame

    # Flush the output memmap to disk
    output_frames.flush()

    print(f"Overlayed video frames saved to {output_video_memmap}")

def display_memmap_frames(memmap_file, num_frames, height, width, channels=3):
    # Load the memmap array with the correct shape
    frames = np.memmap(memmap_file, dtype='uint8', mode='r', shape=(num_frames, height, width, channels))

    for i in range(num_frames):
        # Read the frame from the memmap array
        frame = frames[i]

        # Display the frame (no need to swap channels if they are in BGR format)
        cv2.imshow("Frame", frame)

        # Wait a short duration (0.01 seconds) before showing the next frame
        time.sleep(0.01)

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            break

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    overlay_videos(background_video_memmap='walking_10_first.dat',
                   foreground_video_memmap='walking_10_last.dat',
                   output_video_memmap='overlayed_video.dat',
                   num_frames=250,
                   height=1080,
                   width=1920)

    display_memmap_frames(memmap_file='overlayed_video.dat',
                          num_frames=250,
                          height=1080,
                          width=1920)
