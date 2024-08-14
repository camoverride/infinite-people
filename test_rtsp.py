import cv2



# Replace with the actual RTSP URL of your Amcrest security camera
rtsp_url = "rtsp://admin:admin123@192.168.0.109:554/live"

# Open a connection to the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Display the frame
    cv2.imshow('RTSP Stream', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
