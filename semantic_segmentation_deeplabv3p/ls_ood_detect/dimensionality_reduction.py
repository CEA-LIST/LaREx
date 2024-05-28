from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pacmap
from sklearn.decomposition import PCA


def apply_pca_ds(
    train_samples: np.ndarray,
    test_samples: np.ndarray,
    nro_components: int = 16,
    svd_solver: str = "randomized",
    whiten: bool = True,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to a dataset

    :param train_samples: PCA train and fit samples
    :type train_samples: np.ndarray
    :param test_samples: PCA test samples
    :type test_samples: np.ndarray
    :param nro_components: PCA components number
    :type nro_components: int
    :param svd_solver: PCA SVD solver
    :type svd_solver: str
    :param whiten: PCA whiten value
    :type whiten: bool
    :return: Train and Test samples with reduced dimensionality
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pca_dim_red = PCA(n_components=nro_components, svd_solver=svd_solver, whiten=whiten)
    train_ds = pca_dim_red.fit_transform(train_samples)
    test_ds = pca_dim_red.transform(test_samples)
    return train_ds, test_ds, pca_dim_red


def apply_pca_ds_split(
    samples: np.ndarray,
    nro_components: int = 16,
    svd_solver: str = "randomized",
    whiten: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """
    Applies PCA dimensionality reduction to a dataset split

    :param samples: dataset split samples
    :type samples: np.ndarray
    :param nro_components: PCA nnumber of components, defaults to 16
    :type nro_components: int, optional
    :param svd_solver: PCA SVD solver, defaults to 'randomized'
    :type svd_solver: str, optional
    :param whiten: PCA whiten value, defaults to True
    :type whiten: bool, optional
    :return: dataset samples with reduced dimensionality, PCA transformation object
    :rtype: Tuple[np.ndarray, PCA]
    """
    pca_dim_red = PCA(n_components=nro_components, svd_solver=svd_solver, whiten=whiten)
    dataset_dim_red = pca_dim_red.fit_transform(samples)
    return dataset_dim_red, pca_dim_red


def apply_pca_transform(samples: np.ndarray, pca_transform: PCA) -> np.ndarray:
    samples_dim_red = pca_transform.transform(samples)
    return samples_dim_red


def plot_samples_pacmap(
    samples_ind: np.ndarray,
    samples_ood: np.ndarray,
    neighbors: int = 25,
    components: int = 2,
    title: str = "Plot Title",
    return_figure: bool = False,
) -> Union[None, plt.scatter]:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    :param samples_ind: In-Distribution (InD) samples numpy array
    :type samples_ind: np.ndarray
    :param samples_ood: Out-of-Distribution (OoD) samples numpy array
    :type samples_ood: np.ndarray
    :param neighbors: Number of nearest-neighbors considered for the PaCMAP algorithm
    :type neighbors: int
    :param components: Number of components in final reduction
    :type components: int
    :param title: Plot tile
    :type title:  str
    :param return_figure
    :type return_figure: bool
    :return: Either a plot or None (show the plot)
    :rtype:
    """
    samples_concat = np.concatenate((samples_ind, samples_ood))
    label_normal = np.zeros((samples_ind.shape[0], 1))
    label_anomaly = np.ones((samples_ood.shape[0], 1))
    labels = np.concatenate((label_normal, label_anomaly))
    embedding = pacmap.PaCMAP(n_components=components, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0)
    samples_transformed = embedding.fit_transform(samples_concat, init="pca")

    # visualize the embedding
    # ToDo: Add Axis Names and plot legend
    fig, axes = plt.subplots()
    scatter = axes.scatter(
        samples_transformed[:, 0],
        samples_transformed[:, 1],
        cmap="brg",
        c=labels,
        s=1.5,
    )
    axes.set_title(title)
    axes.legend(
        handles=scatter.legend_elements()[0],
        labels=["In-Distribution", "Out-of-Distribution"],
    )
    if return_figure:
        return fig
    else:
        plt.show()


def fit_pacmap(samples_ind: np.array, neighbors: int = 25, components: int = 2) -> Tuple[np.array, pacmap.PaCMAP]:
    """
    In-Distribution vs Out-of-Distribution Data Projection 2D Plot using PaCMAP algorithm.

    :param components: Number of components in the output
    :type components: int
    :param samples_ind: In-Distribution (InD) samples numpy array
    :type samples_ind: np.ndarray
    :param neighbors: Number of nearest-neighbors considered for the PaCMAP algorithm
    :type neighbors: int
    :return:
    :rtype: None
    """
    embedding = pacmap.PaCMAP(n_components=components, n_neighbors=neighbors, MN_ratio=0.5, FP_ratio=2.0)
    samples_transformed = embedding.fit_transform(samples_ind, init="pca")
    return samples_transformed, embedding


def apply_pacmap_transform(new_samples: np.array, original_samples: np.array, pm_instance: pacmap.PaCMAP) -> np.array:
    return pm_instance.transform(X=new_samples, basis=original_samples)
