import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.transforms as transforms
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from typing import List, Tuple, Union
from PIL import Image


def cross_product_with_dot(vector1: np.ndarray, vector2: np.ndarray):
    # ensures arrays are 2D arrays
    v1 = np.reshape(vector1, (-1, 1))
    v2 = np.reshape(vector2, (1, -1))
    return np.dot(v1, v2)


def cross_product_with_matmul(vector1: np.ndarray, vector2: np.ndarray):
    return np.matmul(vector1[:, np.newaxis], vector2[:, np.newaxis].T)


def cross_product_with_outer(vector1: np.ndarray, vector2: np.ndarray):
    return np.outer(vector1, vector2)


def compute_mean_of_squares(data: np.ndarray):
    # data[0:1] = data[0][:, np.newaxis].T
    my_way = sum([np.matmul(d[:, np.newaxis], d[:, np.newaxis].T) for d in data]) / len(data)
    original_way = sum([np.matmul(data[i:i+1].T, data[i:i+1]) for i in range(len(data))]) / len(data)
    assert np.array_equal(my_way, original_way)
    return my_way


def compute_mean(data: np.ndarray):
    return np.mean(data, axis=0)


def compute_mean_of_squares(data: np.ndarray):
    # data[0:1] = data[0][:, np.newaxis].T
    my_way = sum([np.matmul(d[:, np.newaxis], d[:, np.newaxis].T) for d in data]) / len(data)
    original_way = sum([np.matmul(data[i:i+1].T, data[i:i+1]) for i in range(len(data))]) / len(data)
    assert np.array_equal(my_way, original_way)
    return my_way


def modify_covariance_matrix(covariance_matrix: np.ndarray):
    '''Re-computes the input covariance matrix to prevent numerical instability.'''
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    clipped_eigenvalues = np.maximum(eigenvalues, 1e-3)
    diagonal_matrix = np.diag(clipped_eigenvalues)
    reconstructed_matrix = np.matmul(np.matmul(eigenvectors, diagonal_matrix), np.linalg.inv(eigenvectors))
    return reconstructed_matrix


def modify_covariance_matrix2(covariance_matrix: np.ndarray):
    '''A lighter version.'''
    return covariance_matrix + np.eye(len(covariance_matrix)) * 1e-3


def generate_clustered_data(
    num_dimensions: int,
    num_clusters: int,
    cluster_means: Union[np.ndarray, list],
    num_points_per_cluster: int = 100,
    spread_factor: float = 0.1,
    rng: np.random.Generator = np.random.default_rng(),
    plot: bool = False,
    legend: bool = False
) -> Tuple[np.ndarray, Union[plt.Figure, None], Union[plt.Axes, None]]:
    '''
    Generates clustered data points and optionally plots them in 2D or 3D.

    Parameters:
    -----------
    num_dimensions : int
        Number of dimensions. Only 2 or 3 dimensions are supported.

    num_clusters : int
        Number of clusters to generate.

    cluster_means : Union[np.ndarray, list]
        Means for each cluster.

    num_points_per_cluster : int, optional
        Number of points per cluster. Defaults to 100.

    spread_factor : float, optional
        Spread factor determining the spread of clusters. Defaults to 0.1.

    rng : np.random.Generator, optional
        The random generator to sample data. Defaults to the default numpy generator.

    plot : bool, optional
        Whether to plot the generated data. Defaults to False.

    legend : bool, optional
        Whether to display a legend in the plot. Defaults to False.

    Returns:
    --------
    Tuple[np.ndarray, Union[plt.Figure, None], Union[plt.Axes, None]]
        Tuple containing generated data points, figure (if plotted), and axis (if plotted).
    '''

    if num_dimensions not in [2, 3]:
        raise ValueError('Visualization is only possible for 2D or 3D data.')

    covariances = []
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            dist = np.linalg.norm(np.array(cluster_means[i]) - np.array(cluster_means[j]))
            covariances.append(np.eye(num_dimensions) * dist * spread_factor)

    data = []
    for i in range(num_clusters):
        cluster_data = rng.multivariate_normal(mean=cluster_means[i], cov=covariances[i*(num_clusters-1)//2], size=num_points_per_cluster)
        data.append(cluster_data)

    generated_data = np.concatenate(data)

    fig = None
    ax = None
    if plot:
        fig = plt.figure(figsize=(10, 10))
        fig.set_facecolor('white')
        if num_dimensions == 2:
            ax = fig.add_subplot(111)
            for i in range(num_clusters):
                ax.scatter(generated_data[i * num_points_per_cluster: (i + 1) * num_points_per_cluster, 0],
                           generated_data[i * num_points_per_cluster: (i + 1) * num_points_per_cluster, 1],
                           label='Mean: {}'.format(cluster_means[i]), s=20)
        elif num_dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(num_clusters):
                ax.scatter(generated_data[i * num_points_per_cluster: (i + 1) * num_points_per_cluster, 0],
                           generated_data[i * num_points_per_cluster: (i + 1) * num_points_per_cluster, 1],
                           generated_data[i * num_points_per_cluster: (i + 1) * num_points_per_cluster, 2],
                           label='Mean: {}'.format(cluster_means[i]), s=20)
        if legend:
            ax.legend()
        fig.tight_layout()
    return generated_data, fig, ax


def plot_gaussian(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    color = kwargs.pop('color', None)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    if color is not None:
        ax.plot(mean_x, mean_y, marker='o', markersize=10, markeredgecolor=color, markerfacecolor=color)
    return ax.add_patch(ellipse)


def plot_gaussians(means, covariances, ax, n_std=3.0, rng: np.random.Generator = np.random.default_rng(), **kwargs):
    '''
        Plots multiple Gaussian distributions on a given axis.

    Parameters:
    -----------
    means : list
        List of means for each Gaussian distribution.

    covariances : list
        List of covariance matrices for each Gaussian distribution.

    ax : matplotlib.axes.Axes
        The axis on which the Gaussians will be plotted.

    n_std : float, optional
        Number of standard deviations for contour plotting. Defaults to 3.0.

    rng : np.random.Generator, optional
        Random number generator. Defaults to np.random.default_rng().

     **kwargs: Additional keyword arguments.
        - cmap (LinearSegmentedColormap, optional): The cmap to sample RGBA colors. Defaults to plt.cm.jet.
        - cmap_values (List[float], optional): The values to sample RGBA colors (from the cmap). Defaults to uniform sampling.
    Returns:
    --------
    matplotlib.axes.Axes
        The axis containing the plotted Gaussians.
    '''
    k = len(means)
    assert k == len(covariances)
    cmap = kwargs.pop('cmap', plt.cm.jet) # plt.cm.jet is a LinearSegmentedColormap
    color_values_in_cmap = kwargs.pop('cmap_values', rng.uniform(size=k))
    rgba_colors = [cmap(i) for i in color_values_in_cmap]

    for i in range(k):
        plot_gaussian(means[i], covariances[i], ax, n_std=n_std, edgecolor=rgba_colors[i], color=rgba_colors[i], **kwargs)

    return ax


def remove_patches(ax):
    ax.patches.clear()


def remove_plotted_points(ax):
    to_remove = []
    for artist in ax.get_children():
        if isinstance(artist, plt.Line2D) and len(artist.get_xdata()) == 1:
            to_remove.append(artist)

    for line in to_remove:
        line.remove()


def remove_gaussian(ax):
    '''Utility function that calls remove_patches and remove_plotted_points of the axis.'''
    remove_patches(ax)
    remove_plotted_points(ax)


def log_likelihood(data: np.ndarray, coefficients, means, covariances):
    k = len(coefficients)
    log_likelihoods = []
    for x in data:
        tmp = 0.0
        for i in range(k):
            tmp += coefficients[i] * multivariate_normal.pdf(x, mean=means[i], cov=covariances[i])
        log_likelihoods.append(np.log(tmp))
    return sum(log_likelihoods)


def create_gif(folder_path: str, output_gif_name: str = 'output.gif', duration: int = 100, prefix: str = '') -> None:
    prefix = prefix.lower()
    png_files = [file for file in os.listdir(folder_path) if (file.lower().endswith('.png')) and (file.lower().startswith(prefix))]
    png_files.sort()

    images = []
    for file_name in png_files:
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path)
        images.append(img)

    output_gif_name = output_gif_name.split('.gif')[0] + '.gif'
    images[0].save(
        output_gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration
    )


def plot_points(ax, data: Union[np.ndarray, List[np.ndarray]], **kwargs):
    '''
        Plots a line with markers along it on a given axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which the Gaussians will be plotted.

    data : Union[np.ndarray, list]
        The points to plot.

     **kwargs: Additional keyword arguments.
        - color (str, optional): The color for the line and the markers. Default to blue.
        - step (int, optional): The step for the x ticks. Default to 1.
        - markersize (int, optional): The size of the points. Default to 8.
    Returns:
    --------
    matplotlib.axes.Axes
        The axis containing the plotted data.
    '''
    color = kwargs.get('color', 'blue')
    step = kwargs.get('step', 1)
    markersize = kwargs.get('markersize', 8)
    x = np.arange(len(data))
    ax.plot(x, data, color=color)
    for j, l in enumerate(data):
        ax.plot(j, l, marker='o', markersize=markersize, markeredgecolor=color, markerfacecolor=color)
    ax.set_xticks(np.arange(len(data), step=step))
    return ax