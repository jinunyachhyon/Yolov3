import cv2
import os

# List image files
image_folder = 'test_image'
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # Sort the image file names to ensure correct order

# Set the desired dimensions
desired_width = 416
desired_height = 416

# Set video parameters
frame_width, frame_height = (416, 416)  # Set frame dimensions
out = cv2.VideoWriter('test_video/test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (frame_width, frame_height))

for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)

    # Get the current dimensions of the image
    height, width, _ = img.shape

    # Calculate padding values to reach the desired dimensions
    pad_x = max(0, desired_width - width)
    pad_y = max(0, desired_height - height)

    # Calculate padding on all sides
    top_pad = pad_y // 2
    bottom_pad = pad_y - top_pad
    left_pad = pad_x // 2
    right_pad = pad_x - left_pad

    # Apply padding to the image
    img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize the image to the desired dimensions (416x416)
    img = cv2.resize(img, (desired_width, desired_height))

    out.write(img)

# Release Video object
out.release()