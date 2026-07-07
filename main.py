import cv2 as cv

# constants
KINECT_WIDTH = 640
KINECT_HEIGHT = 480

def resize_frame(frame):
    # scale frame height down to KINECT_HEIGHT, then scale width proportionally
    og_height, og_width = frame.shape[:2]
    scale = max(KINECT_WIDTH / og_width, KINECT_HEIGHT / og_height)
    new_width = int(og_width * scale)
    resized = cv.resize(frame, (new_width, KINECT_HEIGHT), interpolation=cv.INTER_AREA)

    # center crop if too wide for KINECT_WIDTH
    if new_width > KINECT_WIDTH:
        crop_start = (new_width - KINECT_WIDTH) // 2
        return resized[:, crop_start:crop_start + KINECT_WIDTH]

    return resized

def process_frame(frame):
    frame = resize_frame(frame)
    
    return frame

def main():
    # start webcam, then open a window to display the video feed
    cap: cv.VideoCapture = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # show processed webcam feed in a window
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = process_frame(frame)
        cv.imshow('Webcam Feed', frame)
        
        # wait for 1 ms and check if 'q' key is pressed to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()