import cv2
import torch
from torchvision import transforms
from PIL import Image

from model import YOLOv3  
from utils import *  

# Load your YOLOv3 model
model = YOLOv3(num_classes=config.NUM_CLASSES)
model_path = "Yolov3_epoch80.pth"  # Replace with your model path
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load the video
video_path = "test102.mp4"
cap = cv2.VideoCapture(video_path)

# Define an output video writer
output_path = 'output_video.mp4'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = transform(frame)
    input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        out = model(input_tensor)
        batch_size, A, S, _, _ = out[0].shape
        anchor = torch.tensor([*config.ANCHORS[0]]).to(config.DEVICE) * S
        boxes_scale_i = cells_to_bboxes(out[0], anchor, S=S, is_preds=True)
        bboxes = []
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes += box

        nms_boxes = non_max_suppression(
            bboxes, iou_threshold=0.5, threshold=0.6, box_format="midpoint",
        )

        frame = display_video(frame, nms_boxes)

    # Write processed frame to the output video
    out.write(frame)

    # # Display the frame (optional)
    # cv2.imshow('Frame', frame)
    # if cv2.waitKey(2000) & 0xFF == ord('q'):
    #     break

# Release video capture and writer, and close windows
cap.release()
out.release()
cv2.destroyAllWindows()    
