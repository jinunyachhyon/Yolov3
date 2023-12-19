import cv2
import torch
import argparse
from torchvision import transforms
from PIL import Image

from model import YOLOv3  
from utils import *  

def main(model_path, video_path):
    # Load your YOLOv3 model
    model = YOLOv3(num_classes=config.NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Define transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Define an output video writer
    output_path = './inference_video/output_video.mp4'
    frame_width = config.IMAGE_SIZE
    frame_height = config.IMAGE_SIZE
    fps = 1
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(rgb_frame)

        # Apply transformation
        input_tensor = transform(frame_pil).unsqueeze(0).to(config.DEVICE)

        # Perform inference on the image
        with torch.no_grad():
            out = model(input_tensor)
            batch_size, A, S, _, _ = out[0].shape
            anchor = torch.tensor([*config.ANCHORS[0]]).to(config.DEVICE) * S
            boxes_scale_i = cells_to_bboxes(
                out[0], anchor, S=S, is_preds=True
            )
            bboxes = []
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes += box

            nms_boxes = non_max_suppression(
                bboxes, iou_threshold=0.5, threshold=0.1, box_format="midpoint",
            )
            print(nms_boxes)
            bgr_image = draw_bounding_boxes(frame, nms_boxes)

        # Write processed frame to the output video
        video_writer.write(bgr_image)


    # Release video capture and writer, and close windows
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 Object Detection")
    parser.add_argument("--model", type=str, help="Path to the YOLOv3 model")
    parser.add_argument("--test_video", type=str, help="Path to the test video")
    args = parser.parse_args()

    main(args.model, args.test_video)
