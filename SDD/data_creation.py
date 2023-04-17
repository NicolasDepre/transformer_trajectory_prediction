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


img_step = 10
new_size = (256, 256)
box_size = 16

output_folder = f"{scene}/{new_size[0]}_{new_size[1]}_{box_size}"
try:
    os.mkdir(output_folder)
except FileExistsError as e:
    pass

new_annotations = pd.DataFrame(columns=cols)

for ind, row in data.iterrows():
    if (ind % img_step != 0):
        continue
    frame = f"{row.frame:05d}"
    image_path = f"{scene}/frames/{frame}.jpg"
    outname = f"{output_folder}/{row.track_id:03d}_{frame}.jpg"

    x_scale = old_size[0] / new_size[0]
    y_scale = old_size[1] / new_size[1]
    x_center = ((row["xmax"] + row["xmin"]) // 2) / x_scale
    y_center = ((row["ymax"] + row["ymin"]) // 2) / y_scale
    left = round(x_center - (box_size // 2))
    top = round(y_center - (box_size // 2))
    right = round(x_center - 1 + (box_size // 2))
    bottom = round(y_center - 1 + (box_size // 2))

    if not os.path.exists(outname):
        img = Image.open(image_path)
        old_size = img.size
        img = img.convert("L").resize(new_size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)
        draw.rectangle((left, top, right, bottom), fill="black")
        img.save(outname)

    row.xmax = right
    row.xmin = left
    row.ymin = top
    row.ymax = bottom
    new_annotations.loc[ind] = row

new_annotations.to_csv(output_folder + "/annotations_" + str(img_step) + ".txt", sep=" ", index=False)