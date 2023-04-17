import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageDraw

scene = "datasets/bookstore/video0"
annotation_filename = scene + "/annotations.txt"

cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
data = pd.read_csv(annotation_filename, sep=" ", names=cols)

# Parameters of transform
img_step = 5
old_size = (1424, 1088)
new_size = (128, 128)
box_size = 8


# Computation of new annotations
data = data[data.index % img_step == 0]

x_scale = old_size[0] / new_size[0]
y_scale = old_size[1] / new_size[1]

data["x"] = round(((data["xmax"] + data["xmin"]) / 2) / x_scale, 2)
data["y"] = round(((data["ymax"] + data["ymin"]) / 2) / y_scale, 2)

data["xmin"] = round(data["x"] - (box_size/2)).astype(int)
data["xmax"] = round(data["xmin"] + (box_size-1)).astype(int)
data["ymin"] = round(data["y"] - (box_size/2)).astype(int)
data["ymax"] = round(data["ymin"] + (box_size-1)).astype(int)

output_folder = f"{scene}/{new_size[0]}_{new_size[1]}_{box_size}"

try:
    os.mkdir(output_folder)
except FileExistsError as e:
    pass

data.to_csv(output_folder + "/annotations_" + str(img_step) + ".txt", sep=" ", index=False)


for ind, row in data.iterrows():

    frame = f"{row.frame:05d}"
    image_path = f"{scene}/frames/{frame}.jpg"
    outname = f"{output_folder}/{row.track_id:03d}_{frame}.jpg"

    if not os.path.exists(outname):
        img = Image.open(image_path)
        old_size = img.size
        img = img.convert("L").resize(new_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)

        left = row.xmin
        top = row.ymin
        right = row.xmax
        bottom = row.ymax
        draw.rectangle((left, top, right, bottom), fill="black")

        img.save(outname)



























