import argparse
import config
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import YOLOv3
from loss import YoloLoss
from dataset import YOLODataset
from utils import *


def main(model_path, test_loader, loss_fn, scaled_anchors):

    # Load the model
    model = YOLOv3(num_classes=config.NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(config.DEVICE)

    # Testing --> loss and mAP calculation
    losses = []

    with torch.no_grad():
        model.eval()

        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(config.DEVICE)
            y0, y1, y2 = (y[0].to(config.DEVICE),
                        y[1].to(config.DEVICE),
                        y[2].to(config.DEVICE))
            
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

            losses.append(loss.item())

    print(f"Loss: {sum(losses)/len(losses):.4f}")


    pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    
    print(f"MAP: {mapval.item()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 Object Detection")
    parser.add_argument("--model", type=str, help="Path to the YOLOv3 model")
    args = parser.parse_args()

    transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])

    test_dataset = YOLODataset(
        "test.csv",
        transform=transform,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # Loss function
    loss_fn = YoloLoss()

    # Scaled Anchors
    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor([13,26,52]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)).to(config.DEVICE)

    main(args.model, test_loader, loss_fn, scaled_anchors)