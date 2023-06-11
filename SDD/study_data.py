import os
import pandas as pd

sdd = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

all_stats = []
for scene in os.listdir(sdd):
    scene_stats = []
    for video in os.listdir(sdd+scene):
        folder = f"{sdd}{scene}/{video}"
        #print(folder)
        annotation_filename = folder + "/annotations.txt"
        data = pd.read_csv(annotation_filename, sep=" ", names=cols)

        num_pos = len(data)
        num_trajs = len(data['track_id'].unique())
        scene_stats.append((video, len(data), num_trajs))
    all_stats.append((scene, scene_stats))

stat_cols = ['Scene', ['Video', '\# Positions', '\# Trajectories']]

stats = pd.DataFrame(all_stats, columns=stat_cols)#.sort_values(['Scene'])

print(stats)
#print(stats.to_latex(index=False, longtable=True, column_format="|l|r|r|", hlines=True, caption="Statistics of the Stanford Drone Dataset", label="sdd_stats"))

exit(0)
