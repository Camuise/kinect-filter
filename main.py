import cv2 as cv

# constants
X_RESOLUTION = 640
Y_RESOLUTION = 480

def main():
    # start webcam, then open a window to display the video feed
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # display the frame in a window
        cv.imshow('Webcam Feed', frame)
        
        # wait for 1 ms and check if 'q' key is pressed to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        
    # release the webcam and close the window
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()