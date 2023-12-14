import numpy as np
import cv2
import importlib
import random
import time

import sys
import os
import threading
import queue

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))


# Config file
params_path = "params.py"
config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS


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

        read_frame_queue.put(frame)  # Put the frame into the queue   
    cap.release()  # Release the video file when done
    print("Video Read")


def preprocess(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)
    return frame
    

# Execution
def inference(read_frame_queue, batch_detections_queue):
    print("Enter inference")
    time.sleep(2)

    while True:
        if read_frame_queue.empty():
            break

        frame = read_frame_queue.get() 

        preprocessed_frame = preprocess(frame)
        batch_detections_queue.put(preprocessed_frame)



def display_queue(batch_detections_queue):
    print("Inside display")
    time.sleep(3)

    while True:
        if batch_detections_queue.empty():
            break

        im = batch_detections_queue.get()
        
        # Display image
        cv2.imshow("Prediction", im)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def main():
    video_path = "../test_video/test102.mp4"

    # Create shared queue
    read_frame_queue = queue.Queue()
    batch_detections_queue = queue.Queue()

    t1 = threading.Thread(target=read_video, args=(video_path, read_frame_queue))

    t2 = threading.Thread(target=inference, args=(read_frame_queue, batch_detections_queue))

    t3 = threading.Thread(target=display_queue, args=(batch_detections_queue,))

    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    main()
