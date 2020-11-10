import cv2

# img = cv2.imread("/home/hossein/synthesisData/training/masks/image_67001_layer.png", cv2.IMREAD_GRAYSCALE)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)


import shutil
import os

source = "/home/hossein/synthesisData/training/total_masks"
destination = "/home/hossein/synthesisData/training/masks"

files = os.listdir(source)

for file in files:
    image_number = int(file[6:11])
    if (image_number % 10 == 1) or (image_number % 10 == 6):
        new_path = shutil.move(f"{source}/{file}", destination)
