# import cv2

# # Path to the input video
# input_video_path = "walking.mp4"

# # Open the video file
# cap = cv2.VideoCapture(input_video_path)

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frame
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frame
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

# # Calculate the number of frames for the first 10 seconds
# frames_to_save = fps * 10  # 10 seconds of video

# # Define the codec and create VideoWriter object
# output_video_path = "walking_10.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# # Read and write the first 10 seconds of frames
# for i in range(frames_to_save):
#     ret, frame = cap.read()
#     if not ret:
#         break  # Break if the video ends before reaching 10 seconds
#     out.write(frame)

# # Release resources
# cap.release()
# out.release()

# print(f"New video saved as {output_video_path}")

import cv2

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

# Example usage
video_path = "output_video_2.mp4"
width, height, num_frames = get_video_properties(video_path)
print(f"Width: {width}, Height: {height}, Number of Frames: {num_frames}")
