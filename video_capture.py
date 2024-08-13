import cv2
import time

def capture_video(video_path, duration=5, fps=30, width=640, height=480):
    # Open the default camera (usually the first one)
    cap = cv2.VideoCapture(0)
    
    # Set video frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Calculate the number of frames to capture
    num_frames = int(duration * fps)
    
    print(f"Recording {video_path} for {duration} seconds...")
    
    # Capture frames
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        
        # Display the frame (optional)
        cv2.imshow('Recording...', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def capture_background(background_path, width=640, height=480):
    # Open the default camera (usually the first one)
    cap = cv2.VideoCapture(0)
    
    # Set video frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    print("Press 'c' to capture the background image.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Capture Background', frame)
        
        # Wait for the 'c' key to be pressed to capture the background image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(background_path, frame)
            print(f"Background image saved as {background_path}")
            break
    
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

# Capture two 5-second videos
capture_video('video_a.avi')
capture_video('video_b.avi')

# Wait for the user to capture a background image
capture_background('background.jpg')
