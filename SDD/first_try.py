import torch.nn as nn
from torch import Tensor, zeros
from PIL import Image
from torchvision import transforms
import os

convert_tensor = transforms.ToTensor()

scene = "bookstore/"
size = "images/small_bw/"

filenames = os.listdir(scene + size)[:100]

dataset = zeros((len(filenames), 136, 178))

for ind, filename in enumerate(filenames):
    img = Image.open(scene + size + filename)
    dataset[ind] = convert_tensor(img)

print(dataset)
print(dataset.shape)
