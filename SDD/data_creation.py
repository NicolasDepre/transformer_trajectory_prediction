import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageDraw


scene = "bookstore"
video = "0"
filename = scene + "/annotations/video" + video + "/annotations.txt" 
reference_image = scene + "/annotations/video" + video + "/reference.gif" 

cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
data = pd.read_csv(filename, sep=" ", names=cols)

scale_factor = 8

for ind, row in data.iterrows():
    if (ind % 10 != 0):
        continue
    frame = f"{row.frame:05d}"
    image_path = f"{scene}/frames/{frame}.jpg"
    #print(image_path)

    img = Image.open(image_path)
    img = img.convert("L").resize((img.size[0]//scale_factor, img.size[1]//scale_factor), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)

    margin = 0
    left = row["xmin"] - margin
    top = row["ymin"] - margin
    right = row["xmax"] + margin
    bottom = row["ymax"] + margin

    #print(row)
    #print(left, top, right, bottom)
    #print((left, top, right, bottom))
    #print((left//scale_factor, top//scale_factor, right//scale_factor, bottom//scale_factor))
    draw.rectangle((left//scale_factor, top//scale_factor, right//scale_factor, bottom//scale_factor), fill="black")
    outname = f"{scene}/images/small_bw/{row.track_id:03d}_{frame}.jpg"
    #print(outname)
    #image.show()
    img.save(outname)