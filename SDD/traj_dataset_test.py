import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import dataset
from torchvision import transforms

from PIL import Image

class TrajDataset(dataset.Dataset):

    to_tensor = transforms.ToTensor()
    # cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

    def __init__(self, data_folder, n_prev=4, n_next=6, img_step=10, n_traj = 10, device=None):
        self.data_folder = data_folder
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.n_traj = n_traj

        self.src = Tensor()
        self.intermediate = Tensor()
        self.tgt = Tensor()
        self.process_data()
        self.device = device

    def process_data(self):
        raw_data = pd.read_csv(self.data_folder + "/annotations_" + str(self.img_step) + ".txt", sep=" ")
        #print(raw_data)
        #raw_data = raw_data[raw_data.index % self.img_step == 0]
        src = []
        intermediate = []
        tgt = []
        raw_data[["xmin", "ymin", "xmax", "ymax"]] = self.normalize(raw_data[["xmin", "ymin", "xmax", "ymax"]])
        track_ids = raw_data["track_id"].unique()[:self.n_traj]

        for track_id in track_ids:
            print("opening track " + str(track_id))
            traj = raw_data[raw_data["track_id"] == track_id]  # get all positions of track
            memo = {}
            for i in range(len(traj) - self.n_next - self.n_prev):
                # n_prev images used to predict
                x = self.get_n_images_after_i(traj, self.n_prev, i, memo)
                x = self.normalize_img(x)
                src.append(x)
                # images that should be predicted
                x2 = self.get_n_images_after_i(traj, self.n_next, self.n_prev + i, memo)
                intermediate.append(self.normalize_img(x2))
                # bounding boxes inside predicted images
                y = traj.iloc[i + self.n_prev: i + self.n_prev + self.n_next][["xmin", "ymin", "xmax", "ymax"]]  # recuperer le grand truth à prédire
                tgt.append(Tensor(self.normalize_coords(y.values))) # add to ground truth dataset

        self.src = torch.stack(src, dim=0)
        self.intermediate = torch.stack(intermediate, dim=0)
        self.tgt = torch.stack(tgt, dim=0)

    def normalize_img(self, img):
        return img / img.norm(p=2)

    def normalize(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def get_n_images_after_i(self, traj, n, i, memo):
        X = []
        for ind, pos in traj.iloc[i: i+n,:].iterrows():
            track_id = pos["track_id"]
            frame = pos["frame"]
            path = f"{self.data_folder}/{track_id:03d}_{frame:05d}.jpg"
            if path in memo:
                img = memo[path]
            else:
                img = Image.open(f"{self.data_folder}/{track_id:03d}_{frame:05d}.jpg")
                memo[path] = img
            img_tensor = self.to_tensor(img)
            X.append(img_tensor)
        return torch.cat(X)

    def __getitem__(self, item):
        src = self.src[item]
        intermediate = self.intermediate[item]
        tgt = self.tgt[item]
        return {"src": src, "intermediate": intermediate, "tgt": tgt}

    def __len__(self):
        return len(self.src)

    def get_image_size(self):
        return self.src[0].size()[1:]

    def get_dataset_infos(self):
        return {"image_size":self.get_image_size(),
                "n_prev": self.n_prev,
                "n_next": self.n_next}