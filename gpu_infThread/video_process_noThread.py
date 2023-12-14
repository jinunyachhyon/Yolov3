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


def read_preprocess_display(video_path):
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

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Output", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    cap.release()  # Release the webcam when done
    print("Video Read")


def main():
    video_path = '../test_video/test102.mp4'  # Use camera index 0 for the default camera

    read_preprocess_display(video_path)


if __name__ == "__main__":
    main()
