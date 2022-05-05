import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from lib.visualization.image import put_text


def load_image(image_file):
    """
    Loads the image and converts to LAB color space. Return the flatten LAB image and the original image dimensions

    Parameters
    ----------
    image_file (str): The file path to the image

    Returns
    -------
    image_vector (ndarray): The flatten image in LAB color space
    dims (tuple): The original image dimensions
    """
    # Read the image
    # Convert to LAB color space
    # Save original dimensions for later use
    # Flatten image along height and width (not depth)
    # Return flattened image and dimensions
    pass


def quantize_image(image_vector, n_clusters):
    """
    Quantize the image

    Parameters
    ----------
    image_vector (ndarray): The image vector
    k (int): Number of clusters

    Returns
    -------
    quantize_image_vector (ndarray): The quantize image vector
    """
    # Create kmeans object with k clusters
    # Fit the image
    # Index the kmeans.cluster_centers_ with the predicted labels in order to quantize the image
    # Return the quantized image
    pass


def show_quantization(image_vector, shape, n_clusters):
    """
    Show's the quantized image

    Parameters
    ----------
    image_vector (ndarray): The image vector
    shape (tuple): The original image dimensions
    k (int): Number of clusters
    """
    # Reshape the image to original dimensions
    # Convert to BGR color space
    # Show quantized image
    pass


if __name__ == "__main__":
    image_file = "../data/color_segmentation/image.jpg"
    # Load the image and the get original image dimensions
    image, orig_shape = load_image(image_file)

    cluster_range = range(0, 0)  # The range of clusters to test
    assert cluster_range != range(0, 0), "Remember to change the cluster_range"

    for k in reversed(cluster_range):
        # Quantize the image with k clusters
        quant = quantize_image(image, k)
        # Show the quantize image
        show_quantization(quant, orig_shape, k)
