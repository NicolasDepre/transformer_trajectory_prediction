from train import Trainer
from model import SimpleViT
import sys
from traj_dataset import TrajDataset
from torch.utils.data import random_split, DataLoader
from torch.optim import *
from torch.nn import MSELoss
import torch
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--optimizer_name', type=str, default='Adam', help='Name of the optimizer')
    parser.add_argument('--n_next', type=int, default=8, help='Number of next frames to predict')
    parser.add_argument('--n_prev', type=int, default=8, help='Number of previous frames to use for prediction')
    parser.add_argument('--train_prop', type=float, default=0.7, help='Proportion of data to use for training')
    parser.add_argument('--val_prop', type=float, default=0.15, help='Proportion of data to use for validation')
    parser.add_argument('--test_prop', type=float, default=0.15, help='Proportion of data to use for testing')
    parser.add_argument('--img_step', type=int, default=20, help='Frame step size')
    parser.add_argument('--model_dimension', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--patch_size', type=int, default=8, help='Size of the patches')
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input frames')
    parser.add_argument('--block_size', type=int, default=8, help='Size of the block on the images')
    parser.add_argument('--patch_depth', type=int, default=3, help='Number of color channels for patches')
    parser.add_argument('--model_depth', type=int, default=6, help='Depth of the model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads for the multi-head attention')
    parser.add_argument('--mlp_dim', type=int, default=2048, help='Dimension of the MLP in the Transformer')
    parser.add_argument('--dim_head', type=int, default=64, help='Dimension of each head in the multi-head attention')
    parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--teacher_forcing', type=int, default=5, help='Number of epochs where teacher forcing is used')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = parse_args()
    print(args)

    batch_size = args.batch_size
    lr = args.lr
    gpu = args.gpu
    optimizer_name = args.optimizer_name
    n_next = args.n_next
    n_prev = args.n_prev
    train_prop = args.train_prop
    val_prop = args.val_prop
    test_prop = args.test_prop
    img_step = args.img_step
    model_dimension = args.model_dimension
    patch_size = args.patch_size
    img_size = args.img_size
    block_size = args.block_size
    patch_depth = args.patch_depth
    model_depth = args.model_depth
    n_heads = args.n_heads
    mlp_dim = args.mlp_dim
    dim_head = args.dim_head
    n_epoch = args.n_epoch
    teacher_forcing = args.teacher_forcing

    device = device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'

    size = f"{img_size}_{img_size}_{block_size}"
    folders = [f"bookstore/video{k}/" for k in range(6)]
    #folders += [f"coupa/video{k}/" for k in range(4)]
    #folders += [f"quad/video{k}/" for k in range(2)]
    #folders += [f"gates/video{k}/" for k in range(9)]
    #folders += [f"deathCircle/video{k}/" for k in range(5)]
    #folders += [f"little/video{k}/" for k in range(4)]
    #folders += [f"nexus/video{k}/" for k in range(10)]

    n_trajs = [(0, -1) for k in range(len(folders))]
    data_folders = ["/waldo/walban/student_datasets/arfranck/SDD/scenes/" + folder + size for folder in folders]
    
    dataset = TrajDataset(data_folders, n_trajs=n_trajs, n_prev=n_prev, n_next=n_next, img_step=img_step)
    train_data, validation_data, test_data = random_split(dataset, [train_prop, val_prop, test_prop])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = SimpleViT(image_size=img_size, image_patch_size=patch_size, frames=n_prev,
                      frame_patch_size=patch_depth, dim=model_dimension, depth=model_depth, mlp_dim=mlp_dim,
                      device=device, dim_head=dim_head,heads=n_heads)
    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "ADAGRAD":
        optimizer = Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception(f"Optimiser {optimizer} is not handled")
    mse = MSELoss()
    criterion = MSELoss()

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
            "model_dimension": model_dimension,
            "train_length": len(train_loader),
            "val_length": len(val_loader),
            "test_length": len(test_loader),
            "heads": n_heads,
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
