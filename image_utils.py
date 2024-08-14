import glob
import os
import time
import cv2
import numpy as np



def get_most_recent_file(folder_path):
    """
    Get a list of all files in the folder, sorted by creation time
    """
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # If there are no files, return None
    if not files:
        return None
    
    # Get the most recently created file
    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file


class RTSPVideoRecorder:
    def __init__(self, rtsp_url, output_file, duration, fps):
        self.rtsp_url = rtsp_url
        self.output_file = output_file
        self.duration = duration
        self.fps = fps
        self.cap = None
        self.writer = None

    def open_stream(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            raise Exception("Failed to open RTSP stream")

    def setup_writer(self, frame_width, frame_height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (frame_width, frame_height))

    def record(self):
        self.open_stream()
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to read from RTSP stream")

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.setup_writer(frame_width, frame_height)

        start_time = time.time()
        while time.time() - start_time < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, ending recording early")
                break
            self.writer.write(frame)

        self.release_resources()

    def release_resources(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


def get_num_frames(memmap_file, height, width, channels=3):
    """
    Calculate the number of frames based on the size of the memmap file
    """
    file_size = np.memmap(memmap_file, dtype='uint8', mode='r').shape[0]
    frame_size = height * width * channels

    return file_size // frame_size


def display_memmap_frames(memmap_file, height, width, channels=3):
    """
    Display frames from the memmap as a "video"
    """
    # Dynamically calculate the number of frames
    num_frames = get_num_frames(memmap_file, height, width, channels)

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


def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return width, height, num_frames



if __name__ == "__main__":
    rtsp_url = "rtsp://admin:admin123@192.168.0.109:554/live"

    recorder = RTSPVideoRecorder(rtsp_url, output_file="output_video_1.mp4")
    recorder.record()

    recorder = RTSPVideoRecorder(rtsp_url, output_file="output_video_2.mp4")
    recorder.record()
