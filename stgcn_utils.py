import os
import zipfile
import numpy as np
import torch


def load_metr_la_data():#对X进行预处理
    if (not os.path.isfile("data/adj_mat.npy")
            or not os.path.isfile("data/node_values.npy")):
        with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("data/adj_mat.npy")
    X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def calSimilarityByDistance(C=40):
    file='./adjacent_A.npy'
    adjacencyMatrix=np.load(file)

    # meanValue = np.mean(adjacencyMatrix)
    # initialAdjacency = np.zeros((C, C))
    # for i in range(C):
    #     for j in range(C):
    #         if adjacencyMatrix[i][j] > meanValue:
    #             initialAdjacency[i][j] = 1

    return adjacencyMatrix





def load_traffic_data(X_dir,label_dir):#对X进行预处理
    A=calSimilarityByDistance(390)
    X = np.load(X_dir)
    label=np.load(label_dir)


    # return A, X, np.array(label_one_hot), means, stds
    labels=[]
    for i in label:
        a=np.argwhere(i)
        if len(a)==0:
            labels.append(0)
        else:
            labels.append(a[0][0]+1)
    labels=np.array(labels).astype('int64')
    labels_anomaly = np.where(labels > 0)[0]

    # label_unique=np.unique(labels)

    labels_normal=np.where(labels==0)[0]
    labels_normal=labels_normal[0:len(labels_anomaly)]

    labels_total=np.concatenate((labels_normal,labels_anomaly))
    labels=labels[labels_total]
    X=X[labels_total]

    X = X.astype(np.float32)
    X = np.expand_dims(X, 3).repeat(3, axis=3)
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 1, 3))
    X = X - means.reshape(1, 1, -1, 1)
    stds = np.std(X, axis=(0, 1, 3))
    X = X / stds.reshape(1, 1, -1, 1)
    X = X.transpose(0, 1, 3, 2)

    return A, X,labels, means, stds



def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points


    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def generate_dataset_new(X,label,A):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points


    return torch.from_numpy(X), \
           torch.from_numpy(label),torch.from_numpy(A)
