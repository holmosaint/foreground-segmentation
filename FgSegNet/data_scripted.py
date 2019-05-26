import os
import subprocess
import random

image_dir = "./keras-aug/orderedImages"
gt_dir = "./keras-aug/orderedTruths"
low_list = [24, 28, 91, 145, 206, 223, 231, 235, 269, 280, 308, 316, 324, 329, 343, 344, 350, 361, 426, 429, 430, 445, 461, 511, 512, 543, 545, 580, 608, 622, 629, 634, 636, 651, 729, 749, 800, 806, 814, 847, 900, 932, 939, 951, 955, 988, 995]
low_list = [x + 30000 for x in low_list]
# image_name = [str(x) + ".jpg" for x in low_list]
# gt_name = [str(x) + "_gt.bmp" for x in low_list]

image_name = list()
for path, _dir, file_name in os.walk(image_dir):
    image_name.extend(file_name)
    image_name = [x for x in image_name if x.endswith(".jpg")]

gt_name = list()
for path, _dir, file_name in os.walk(gt_dir):
    gt_name.extend(file_name)
    gt_name = [x for x in gt_name if x.endswith(".bmp")]


random.shuffle(image_name)
print("Train image")
for image in image_name[:900]:
    image = image.split(".")[0]
    path = os.path.join(image_dir+"/train", image)
    folder = os.path.exists(path)
    print(path, end="\r")
    if not folder:
        os.makedirs(path)
    cmd = "mv " + image_dir + "/" + image + ".jpg " + path
    subprocess.call(cmd, shell=True)

print("\nVal image")
for image in image_name[900:]:
    image = image.split(".")[0]
    path = os.path.join(image_dir+"/val", image)
    folder = os.path.exists(path)
    print(path, end="\r")
    if not folder:
        os.makedirs(path)
    cmd = "mv " + image_dir + "/" + image + ".jpg " + path
    subprocess.call(cmd, shell=True)
print(image_name)
print("\nTrain mask")
for mask in image_name[:900]:
    mask = mask.split(".")[0]
    path = os.path.join(gt_dir + "/train", mask+"_gt")
    folder = os.path.exists(path)
    print(path, end="\r")
    if not folder:
        os.makedirs(path)
    cmd = "mv " + gt_dir + "/" + mask + "_gt.bmp " + path
    subprocess.call(cmd, shell=True)


print("\nVal mask")
for mask in image_name[900:]:
    mask = mask.split(".")[0]
    path = os.path.join(gt_dir + "/val", mask+"_gt")
    folder = os.path.exists(path)
    print(path, end="\r")
    if not folder:
        os.makedirs(path)
    cmd = "mv " + gt_dir + "/" + mask + "_gt.bmp " + path
    subprocess.call(cmd, shell=True)

