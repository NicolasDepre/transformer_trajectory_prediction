import os
import pandas as pd

sdd = "/waldo/walban/student_datasets/arfranck/SDD/scenes/"
cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

all_stats = []
index = []
for scene in os.listdir(sdd):
    scene_stats = []
    for video in os.listdir(sdd+scene):
        folder = f"{sdd}{scene}/{video}"
        #print(folder)
        annotation_filename = folder + "/annotations.txt"
        data = pd.read_csv(annotation_filename, sep=" ", names=cols)

        video = int(video[5:])

        num_pos = len(data)
        num_trajs = len(data['track_id'].unique())
        prop_ped = len(data[data['label']=="Pedestrian"]) /num_pos * 100
        prop_bike = len(data[data['label']=="Biker"]) /num_pos * 100
        prop_other = len(data[(data['label'] != "Pedestrian") & (data['label'] != "Biker")]) / num_pos * 100
        all_stats.append((scene, video, len(data), num_trajs, prop_ped, prop_bike, prop_other))
        index.append((scene, video))
    #all_stats.append((scene, scene_stats))

stat_cols = ['Scene', 'Video', '\# Positions', '\# Trajectories', 'Prop pedestrians', 'Prop Bikes', 'Prop other']

#stats = pd.DataFrame(all_stats, columns=stat_cols)#.sort_values(['Scene'])

index = pd.MultiIndex.from_tuples(index, names=['Scene', 'Video'])
stats = pd.DataFrame(index=index, columns=stat_cols[2:])


for scene, video, *infos in all_stats:
    stats.loc[(scene, video), stat_cols[2:]] = infos

#stats = stats.sort_values(["Prop pedestrians", "Scene", "Video"])
stats = stats.sort_values(["Scene", "Video"])
print(stats)
#stats.to_csv('videos_sizes.csv')
print(stats.to_latex(bold_rows=True, longtable=True, multirow=True, column_format='|l|c|r|r|r|r|r|', caption="Statistics of the Stanford Drone Dataset", label="tab:full-sdd-stats"))

exit(0)
