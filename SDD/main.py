import transformer as tf
from traj_dataset import TrajDataset

if __name__ == '__main__':

    model = tf.TrajTransformer(d_model = 128)
    dataset = TrajDataset("datasets/bookstore/video0")

    print(dataset.X.shape)


