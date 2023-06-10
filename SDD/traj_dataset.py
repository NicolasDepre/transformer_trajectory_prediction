
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import dataset, DataLoader, Subset
from torchvision import transforms

from PIL import Image

class TrajDataset(dataset.Dataset):

    to_tensor = transforms.ToTensor()
    # cols = ["track_id", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

    def __init__(self, data_folders, n_prev, n_next, img_step, n_trajs):

        self.data_folders = data_folders
        self.n_prev = n_prev
        self.n_next = n_next
        self.img_step = img_step
        self.n_trajs = n_trajs

        self.src = Tensor()
        self.coords = Tensor()
        self.tgt = Tensor()
        self.process_data()
        self.block_size = int(data_folders[0].split("_")[-1])

    def process_data(self):
        src = []
        coords = []
        tgt = []

        for folder, (first_traj, last_traj) in zip(self.data_folders, self.n_trajs):
            raw_data = pd.read_csv(folder + "/annotations_" + str(self.img_step) + ".txt", sep=" ")
            # print(raw_data)
            # raw_data = raw_data[raw_data.index % self.img_step == 0]

            track_ids = raw_data["track_id"].unique()[first_traj:last_traj]

            for track_id in track_ids:
                print("opening track " + str(track_id) + " from " + folder)
                traj = raw_data[raw_data["track_id"] == track_id]  # get all positions of track
                memo = {}
                for i in range(len(traj) - self.n_next - self.n_prev):
                    # n_prev images used to predict
                    x = self.get_n_images_after_i(folder, traj, self.n_prev, i, memo)
                    src.append(x)
                    # images that should be predicted
                    c = traj.iloc[i: i + self.n_prev][["x", "y"]]
                    coords.append(Tensor(c.values))
                    # bounding boxes inside predicted images
                    y = traj.iloc[i + self.n_prev: i + self.n_prev + self.n_next][["x", "y"]]  # recuperer le grand truth à prédire
                    tgt.append(Tensor(y.values)) # add to ground truth dataset

        self.src = torch.stack(src, dim=0)
        self.coords = self.normalize_coords(torch.stack(coords, dim=0))
        self.tgt = self.normalize_coords(torch.stack(tgt, dim=0))

    def normalize_img(self, img):
        return img

    def normalize_coords(self, tgt):
        return tgt / self.get_image_size()[0]

    def get_n_images_after_i(self, folder, traj, n, i, memo):
        X = []
        for ind, pos in traj.iloc[i: i+ n, :].iterrows():
            track_id = pos["track_id"]
            frame = pos["frame"]
            path = f"{folder}/{track_id:03d}_{frame:05d}.jpg"
            if path in memo:
                img = memo[path]
            else:
                img = Image.open(f"{folder}/{track_id:03d}_{frame:05d}.jpg")
                memo[path] = img
            img_tensor = self.to_tensor(img)
            X.append(self.normalize_img(img_tensor))
        return torch.cat(X)

    def __getitem__(self, item):
        return {
            'src': self.src[item],
            'coords': self.coords[item],
            'tgt': self.tgt[item]
        }

    def __len__(self):
        return len(self.src)

    def get_image_size(self):
        return self.src[0].size()[1:]

    def get_dataset_infos(self):
        return {"image_size": self.get_image_size(),
                "n_prev": self.n_prev,
                "n_next": self.n_next,
                "block_size": self.block_size
                }

