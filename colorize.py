import numpy as np
import argparse
import cv2
import os
import pyautogui
"""
Download the model files: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

"""
# Define default path
DEFAULT_IMAGE = os.path.join("images", "default.jpg")
# Load model paths
DIR = r"C:\Users\RamYa\OneDrive\Desktop\Colorizing"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")
# Argparser with optional image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input black and white image")
args = vars(ap.parse_args())
image_path = args["image"] if args["image"] else DEFAULT_IMAGE
# Validate image file existence
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")
# Load model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
# Set model blobs
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
# Read and preprocess image
image = cv2.imread(image_path)
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50
# Colorize
print("Colorizing image...")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")
# Resize display for screen size
screen_w, screen_h = pyautogui.size()
scale_factor = min(screen_w / image.shape[1], screen_h / image.shape[0], 1)
new_w = int(image.shape[1] * scale_factor)
new_h = int(image.shape[0] * scale_factor)
resized_original = cv2.resize(image, (new_w, new_h))
resized_colorized = cv2.resize(colorized, (new_w, new_h))
# Show output
cv2.imshow("Colorized (Scaled)", resized_colorized)
cv2.imshow("Original (Scaled)", resized_original)
cv2.waitKey(0)
cv2.destroyAllWindows()