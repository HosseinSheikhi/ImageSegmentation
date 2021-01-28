import cv2

# img = cv2.imread("/home/hossein/synthesisData/training/masks/image_67001_layer.png", cv2.IMREAD_GRAYSCALE)
#
# cv2.imshow("img", img)
# cv2.waitKey(0)


import shutil
import os

source = "/home/hossein/synthesisData/training/total_images"
destination = "/home/hossein/synthesisData/training/images"

files = os.listdir(source)

for file in files:
    image_number = int(file[6:11])
    if image_number % 3 == 0:
        new_path = shutil.move(f"{source}/{file}", destination)
