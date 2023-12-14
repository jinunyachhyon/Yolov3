import torch
import torch.nn as nn
import numpy as np
import cv2
import importlib
import random
import time

import sys
import os
from multiprocessing import Queue, Process
import queue

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

from nets.model_main import ModelMain
from gpu_utils import YOLOPost, non_max_suppression


def read_video_frames(video_path, read_frames_queue, original_frames_queue):
    print("Read video..")
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if there are no more frames
        
        # Put the frame into the queue 
        read_frames_queue.put(frame)   
        original_frames_queue.put(frame)

    cap.release()  # Release the video file when done
    print("Video Read")


def preprocess_frame(frame, config):
    print("Preprocess frame")
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)

    # print(image.shape) # (3,416,416)

    image = torch.from_numpy(image)
    image = image.unsqueeze(0)

    # out = model(image)

    # print(image.shape) # ([1,3,416,416])

    print("Preprocess completed")

    return image


# Execution
def inference(read_frames_queue, batch_detections_queue, model, yolo_losses, config):
    print("Execution Video")

    while True:
        if read_frames_queue.qsize() != 0:
           
            frame = read_frames_queue.get()

            start_time = time.time()  # Record the start time
            preprocessed_frame = preprocess_frame(frame, config)

            out = model(preprocessed_frame)

            print(out[0].shape) # ([1,255,13,13]) -> torch

            # Convert tensor to numpy
            output = []
            output.append(out[0].numpy())
            output.append(out[1].numpy())
            output.append(out[2].numpy())
            # print(len(output))
            # print(output[0].shape) # (1, 255, 13, 13) -> numpy

            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i].forward(output[i]))
            output_con = np.concatenate(output_list, 1)

            batch_detections = non_max_suppression(output_con, config["yolo"]["classes"],
                                                    conf_thres=config["confidence_threshold"],
                                                    nms_thres=0.45)

            end_time = time.time()
            frame_execution_time = end_time - start_time
            print(f"Execution time: {frame_execution_time} sec")

            batch_detections_queue.put(batch_detections)



######## Plot prediction with bounding box
def display(batch_detections_queue, original_frames_queue):
    print("display")

    classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]
    # print(classes)

    while True:
        if batch_detections_queue != 0 and original_frames_queue != 0:    
            batch_detections = batch_detections_queue.get()
            frame = original_frames_queue.get()

            for idx, detections in enumerate(batch_detections):
                if detections is not None:
                    im = frame
                    # print(im.shape) # eg. (428, 640, 3)
                    unique_labels = np.unique(detections[:, -1])
                    n_cls_preds = len(unique_labels)
                    bbox_colors = {int(cls_pred): (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls_pred in unique_labels}
                    
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        color = bbox_colors[int(cls_pred)]

                        # Rescale coordinates to original dimensions
                        ori_h, ori_w, _ = im.shape
                        pre_h, pre_w = config["img_h"], config["img_w"]
                        box_h = ((y2 - y1) / pre_h) * ori_h
                        box_w = ((x2 - x1) / pre_w) * ori_w
                        y1 = (y1 / pre_h) * ori_h
                        x1 = (x1 / pre_w) * ori_w

                        # Create a Rectangle patch
                        cv2.rectangle(im, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)), color, 2)

                        # Add label
                        label = classes[int(cls_pred)]
                        cv2.putText(im, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Display image
                    cv2.imshow("Prediction", im)
                    cv2.waitKey(50)



def main():
    video_path = "../test_video/output_video.webm"

    terminate = False

    ####### Load the model -- config , data parallel, restore pretrain model
    params_path = "params.py"
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    model = ModelMain(config, is_training=False)
    model.train(False)

    # Set data parallel
    model = nn.DataParallel(model)

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        state_dict = torch.load(config["pretrain_snapshot"], map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # print(model)


    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        # yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
        #                             config["yolo"]["classes"], (config["img_w"], config["img_h"])))

        yolo_losses.append(YOLOPost(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    while not terminate:
        # Create shared queue
        read_frames_queue = Queue()
        batch_detections_queue = Queue()
        original_frames_queue = Queue()

        p1 = Process(target=read_video_frames, args=(video_path, read_frames_queue, original_frames_queue))

        p2 = Process(target=inference, args=(read_frames_queue, batch_detections_queue, model, yolo_losses, config))

        # p3 = Process(target=display, args=(batch_detections_queue, original_frames_queue))

        p1.start()
        p2.start()
        # p3.start()

        p1.join()
        p2.join()
        # p3.join()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            terminate = True
            p1.terminate()
            p2.terminate()
            # p3.terminate()
      

        if not p1.is_alive() and not p2.is_alive() and not p3.is_alive():
            terminate = True
            p1.terminate()
            p2.terminate()
            # p3.terminate()


if __name__ == "__main__":
    main()
