import cv2
import importlib

import sys
import os
from multiprocessing import Queue, Process

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))


# Config file
params_path = "params.py"
config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS


# Terminating condition
stop_flag = False


def read_video(video_path, read_frame_queue):
    print("Read video..")
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Loop through the frames of the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if there are no more frames

        if stop_flag:
            break

        read_frame_queue.put(frame)  # Put the frame into the queue   
    cap.release()  # Release the video file when done
    print("Video Read")


def preprocess(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)
    frame = cv2.GaussianBlur(frame, (3, 3), 3.0)
    frame = cv2.Canny(frame, 50, 150)

    return frame
    

# Execution
def inference(read_frame_queue, batch_detections_queue):
    print("Enter inference")

    while not stop_flag:
        try:
            frame = read_frame_queue.get() 

            preprocessed_frame = preprocess(frame)
            batch_detections_queue.put(preprocessed_frame)
        except Queue.empty:
            pass



def display_queue(batch_detections_queue):
    print("Inside display")

    while not stop_flag:
        try:
            im = batch_detections_queue.get()
            
            # Display image
            cv2.imshow("Prediction", im)
            cv2.waitKey(1)
        except Queue.empty:
            pass

    cv2.destroyAllWindows()


def main():
    global stop_flag

    video_path = "../test_video/test102.mp4"

    # Create shared queue
    read_frame_queue = Queue()
    batch_detections_queue = Queue()

    p1 = Process(target=read_video, args=(video_path, read_frame_queue))

    p2 = Process(target=inference, args=(read_frame_queue, batch_detections_queue))

    p3 = Process(target=display_queue, args=(batch_detections_queue,))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    
    p3.join()


if __name__ == "__main__":
    main()
