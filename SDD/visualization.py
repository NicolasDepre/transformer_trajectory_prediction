import pandas as pd
import turtle
import matplotlib.pyplot as plt
import numpy as np


scene = "bookstore"
video = "0"
filename = scene + "/annotations/video" + video + "/annotations.txt" 
reference_image = scene + "/annotations/video" + video + "/reference.gif" 

cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
data = pd.read_csv(filename, sep=" ", names=cols)

data["x"] = (data["xmax"] + data["xmin"]) // 2
data["y"] = (data["ymax"] + data["ymin"]) // 2

print(data.head())
print(data.shape)

exit(0)
def plot_track(track_ids):
    # Load the image
    image = plt.imread(reference_image)
    # Plot the image and the trajectory
    fig, ax = plt.subplots()
    ax.imshow(image)

    colors = {"Biker":"r", "Pedestrian":"b"}
    for track_id in track_ids:
        track = data[data["track_id"] == track_id]

        offset_x = pd.concat([track["x"].iloc[1:], track["x"].iloc[-1:]], axis=0)
        offset_x.index = track.index
        offset_y = pd.concat([track["y"].iloc[1:], track["y"].iloc[-1:]], axis=0)
        offset_y.index = track.index

        track["x_dir"] = offset_x - data["x"]
        track["y_dir"] = data["y"] - offset_y

        # Define the trajectory (in pixel coordinates)
        color = colors[track["label"].iloc[0]]

        ax.plot(track["x"], track["y"], marker='.', color=color)
        #ax.quiver(track["x"][::10], track["y"][::10], track["x_dir"][::10], track["y_dir"][::10], color=color)
        
    plt.show()

plot_track(list(range(15)))
