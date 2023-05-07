from train import Trainer
from model import SimpleViT
import sys
from traj_dataset import TrajDataset
from torch.utils.data import random_split, DataLoader
from torch.optim import *
from torch.nn import MSELoss
import torch

"""
Args:
    1)  batch size
    2)  learning rate
    3)  gpu_id 
    4)  optimizer ["adam","SGD"]
    5)  n_next
    6)  n_prev
    7)  train_prop
    8)  val_prop
    9)  test_prop
    10) img_step
    11) model_dimension
    12) patch_size (tuple like (16,16)) 
    13) img_size (tuple like (64,64))
    14) patch_depth
    15) model_depth
    16) n_heads
    17) mlp_dim 
    18) dim_head
    19) n_epoch
    20) teacher_forcing
"""

if __name__ == "__main__":

    args = sys.argv
    batch_size = int(args[1])
    lr = float(args[2])
    gpu = args[3]
    optimizer_name = args[4]
    n_next = int(args[5])
    n_prev = int(args[6])
    train_prop = float(args[7])
    val_prop = float(args[8])
    test_prop = float(args[9])
    img_step = int(args[10])
    model_dimension = int(args[11])
    patch_size = eval(args[12])
    img_size = eval(args[13])
    patch_depth = int(args[14])
    model_depth = int(args[15])
    n_heads = int(args[16])
    mlp_dim = int(args[17])
    dim_head = int(args[18])
    n_epoch = int(args[19])
    teacher_forcing = int(args[20])

    device = device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    size = "64_64_8"
    folders = ["bookstore/video0/"]
    n_trajs = [(0, 5)]
    data_folders = ["datasets/" + folder + size for folder in folders]

    dataset = TrajDataset(data_folders, n_trajs=n_trajs, n_prev=n_prev, n_next=n_next, img_step=img_step)
    train_data, validation_data, test_data = random_split(dataset, [train_prop, val_prop, test_prop])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = SimpleViT(image_size=img_size, image_patch_size=patch_size, frames=n_prev,
                      frame_patch_size=patch_depth, dim=model_dimension, depth=model_depth, mlp_dim=mlp_dim,
                      device=device, dim_head=dim_head)
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "ADAGRAD":
        optimizer = Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception(f"Optimiser {optimizer} is not handled")
    mse = MSELoss()
    criterion = lambda x,y: torch.rsqrt(mse(x,y))

    wandb_config = {
            "device": device,
            "train_prop": train_prop,
            "test_prop": test_prop,
            "val_prop": val_prop,
            "lr": lr,
            "optimizer": optimizer_name,
            "epochs": n_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "teacher_forcing": teacher_forcing,
            "image_size": img_size,
            "patch_size": patch_size,
            "n_prev": n_prev,
            "n_next": n_next,
            "model_dimension": model_dimension
        }
    configuration = {

        "model": model,
        "device": device,
        "train_data": train_loader,
        "test_data": test_loader,
        "val_data": val_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "epochs": n_epoch,
        "lr": lr,
        "teacher_forcing": teacher_forcing,
        "wandb_config": wandb_config
    }

    trainer = Trainer(**configuration)
    trainer.train()
