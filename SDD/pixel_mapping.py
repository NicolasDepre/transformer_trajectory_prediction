import os
from PIL import Image

folder = "bookstore/images/"
size = "small/"
images = os.listdir(folder + size)

print(images[:3])
img = Image.open(folder + size + images[0])
img = img.convert("L").resize((178, 136), Image.Resampling.LANCZOS)

out_size = "small_bw/"
img.save(folder + out_size + images[0])
