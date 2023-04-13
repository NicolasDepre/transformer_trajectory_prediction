import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torchvision import transforms

from PIL import Image

class TrajDataset(dataset.Dataset):

    to_tensor = transforms.ToTensor()
    cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

    def __init__(self, data_folder, n_prev=4, n_next=2, img_step = 10):
        self.data_folder = data_folder
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.X = Tensor()
        self.Y = Tensor()
        self.process_data()


    def process_data(self):
        raw_data = pd.read_csv(self.data_folder + "/annotations.txt", sep=" ", names=self.cols)
        print(raw_data)
        raw_data = raw_data[raw_data.index % self.img_step == 0]
        X = []
        Y = []
        track_ids = raw_data["track_id"].unique()[:5]
        
        for track_id in track_ids:
            print("opening track " + str(track_id))
            traj = raw_data[raw_data["track_id"] == track_id]  # get all positions of track
            # TODO OPTI ouverture images
            memo = {}
            for i in range(len(traj) - self.n_next - self.n_prev):
                x = self.get_n_images_after_i(traj, self.n_prev, i,memo)
                X.append(x)  # add to dataset
                y = traj.iloc[i + self.n_prev: i + self.n_prev + self.n_next][["xmin", "ymin", "xmax", "ymax"]]  # recuperer le grand truth à prédire
                Y.append(Tensor(y.values)) # add to grand truth dataset
        self.X = torch.stack(X,dim=0)
        self.Y = torch.stack(Y,dim=0)

        print(len(X))
        print(X[0].shape)
        print(self.X.shape)
        print(self.Y.shape)

    def get_image_data(self, trajs):
        X_traj = Tensor()
        for t in trajs.itterows():
            track_id = t["track_id"]
            frame = t["frame"]
            img = Image.open(f"{self.data_folder}/images/small_bw")
        return 

    def get_n_images_after_i(self, traj, n, i,memo):
        X = []
        for ind, pos in traj.iloc[i: i+n,:].iterrows():
            track_id = pos["track_id"]
            frame = pos["frame"]
            path = f"{self.data_folder}/images/small_bw/{track_id:03d}_{frame:05d}.jpg"
            if path in memo:
                img = memo[path]
            else:
                img = Image.open(f"{self.data_folder}/images/small_bw/{track_id:03d}_{frame:05d}.jpg")
                memo[path] = img
            img_tensor = self.to_tensor(img)
            X.append(img_tensor)
        return torch.cat(X)
    
    def __getitem__(self, item):
        X = self.X[item]
        Y = self.Y[item]
        return {"X": X, "Y" : Y}

    def __len__(self):
        return len(self.X)


