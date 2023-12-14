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


# Create a global flag to signal the threads to stop
stop_flag = False


def set_flag_after(seconds = 20):
    global stop_flag
    
    time.sleep(seconds)
    stop_flag = True


def read_video(video_path, read_frame_queue):
    global stop_flag
    print("Read video..")
    # Open the webcam (you can specify the camera index, usually 0 for the default camera)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if there are no more frames

        if stop_flag:
            break  # If the stop flag is set, break out of the loop

        read_frame_queue.put(frame)  # Put the frame into the queue  

    cap.release()  # Release the webcam when done
    print("Video Read")


def preprocess(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)
    return frame
    

# Execution
def inference(read_frame_queue, batch_detections_queue):
    print("Enter inference")
    while not stop_flag:
        try:
            frame = read_frame_queue.get(timeout=1)  # Wait for 1 second for a frame in the queue

            # Perform inference or any other processing on the frame here
            preprocessed_frame = preprocess(frame)

            # Put the processed frame into the batch_detections_queue
            batch_detections_queue.put(preprocessed_frame)
        except queue.Empty:
            pass  # If the queue is empty, continue waiting

    print("Exit inference")



def display_queue(batch_detections_queue):
    print("Inside display")
    while not stop_flag:
        try:
            im = batch_detections_queue.get(timeout=1)  # Wait for 1 second for an item in the queue
            # Display image
            cv2.imshow("Prediction", im)
            cv2.waitKey(1)
        except queue.Empty:
            pass  # If the queue is empty, continue waiting

    cv2.destroyAllWindows()
    print("Exit display")


def main():
    video_path = 0  # Use camera index 0 for the default camera

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
    # Start a timer thread to set the stop_flag after 20 seconds
    timer_thread = threading.Thread(target=set_flag_after, args=(20,))
    timer_thread.start()
    
    main()  # Start the main program
    
    # Wait for the timer thread to finish
    timer_thread.join()
