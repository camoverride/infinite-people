import numpy as np
import cv2



def get_num_frames(memmap_file, height, width, channels=3):
    # Calculate the number of frames based on the size of the memmap file
    file_size = np.memmap(memmap_file, dtype='uint8', mode='r').shape[0]
    frame_size = height * width * channels
    return file_size // frame_size


def stream_and_overlay(rtsp_url, overlay_memmap_file, height, width, channels=3):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Load the memmap file with overlay frames
    num_overlay_frames = get_num_frames(overlay_memmap_file, height, width, channels)
    overlay_frames = np.memmap(overlay_memmap_file, dtype='uint8', mode='r', shape=(num_overlay_frames, height, width, channels))

    overlay_index = 0

    while True:
        # Read a frame from the RTSP stream
        ret, rtsp_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from RTSP stream.")
            break

        # Resize the RTSP frame to match the overlay size if necessary
        # rtsp_frame = cv2.resize(rtsp_frame, (width, height))

        # Get the current overlay frame
        overlay_frame = overlay_frames[overlay_index]

        # Create a binary mask from the overlay by checking where there are non-black pixels
        overlay_mask = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2GRAY)
        _, overlay_mask = cv2.threshold(overlay_mask, 1, 255, cv2.THRESH_BINARY)

        # Invert the mask to create a mask for the RTSP frame
        rtsp_mask = cv2.bitwise_not(overlay_mask)

        # Apply the masks
        rtsp_frame_masked = cv2.bitwise_and(rtsp_frame, rtsp_frame, mask=rtsp_mask)
        overlay_frame_masked = cv2.bitwise_and(overlay_frame, overlay_frame, mask=overlay_mask)

        # Combine the frames
        combined_frame = cv2.add(rtsp_frame_masked, overlay_frame_masked)

        # Display the combined frame
        cv2.imshow("Combined Frame", combined_frame)

        # Increment the overlay index and loop back to the start if necessary
        overlay_index = (overlay_index + 1) % num_overlay_frames

        # Break the loop if ESC is pressed
        if cv2.waitKey(1) == 27:  # 27 is the ASCII code for ESC
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    rtsp_url = "rtsp://admin:admin123@192.168.0.109:554/live"
    overlay_memmap_file = "overlayed_video_a.dat"
    height = 1440
    width = 2560

    stream_and_overlay(rtsp_url, overlay_memmap_file, height, width)
