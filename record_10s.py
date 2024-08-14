import cv2
import time

class RTSPVideoRecorder:
    def __init__(self, rtsp_url, output_file, duration=10, fps=30):
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

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:admin123@192.168.0.109:554/live"

    recorder = RTSPVideoRecorder(rtsp_url, output_file="output_video_1.mp4")
    recorder.record()

    recorder = RTSPVideoRecorder(rtsp_url, output_file="output_video_2.mp4")
    recorder.record()
