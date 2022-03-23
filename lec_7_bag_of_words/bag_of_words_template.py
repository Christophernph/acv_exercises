from cgi import test
import os
from tkinter import image_names

import cv2 as cv
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib.visualization.image import put_text


class BoW:
    def __init__(self, n_clusters, n_features):
        # Create the ORB detector
        self.extractor = cv.ORB_create(nfeatures=n_features)
        self.n_clusters = n_clusters
        self.n_features = n_features
        # Make a kmeans cluster
        self.kmeans = KMeans(self.n_clusters, verbose=0)
        params = dict(max_iter=300, n_init=10, tol=1e-4)
        self.kmeans = self.kmeans.set_params(**params)

    def train(self, imgs):
        """
        Make the bag of "words"

        Parameters
        ----------
        imgs (list): A list with training images. Shape (n_images)
        """
        # Compute the descriptors for each training img
        # Concatenate the list of lists of descriptors to create a list of descriptors
        # Create a KMeans object with n clusters, and fit the list of descriptors
        # Use the hist function to create a histogram database by calculating the histogram for each img
        # (i.e. call the hist function on each list of descriptors in the list of lists)
        # Hint -- save class variables like 'self.object = create_object()' in order to use them in other functions
        
        n = len(imgs)
        
        # Compute features
        features = []
        for i in range(n):
            _, des = self.extractor.detectAndCompute(imgs[i], None)
            features.append(des)
        features = np.vstack(features)

        # Cluster using kmeans
        labels = self.kmeans.fit_predict(features)
        labels = labels.reshape((n, -1))
        self.min_hist = np.min(labels)
        self.max_hist = np.max(labels)
        
        self.histograms = []
        score = np.zeros((self.n_clusters, ))
        for i in range(n):
            hist, _ = np.histogram(labels[i], self.n_clusters, (self.min_hist,  self.max_hist), density=False)
            hist = np.float64(hist) / self.n_features
            self.histograms.append(hist)

            # Count Ni
            score += np.int64(hist > 0)
        self.histograms = np.vstack(self.histograms)
        
        # Inverse document score
        self.idf = np.log(n / score)


    def hist(self, descriptors):
        """
        Make the histogram for words in the descriptors

        Parameters
        ----------
        descriptors (ndarray): The ORB descriptors. Shape (n_features, 32)

        Returns
        -------
        hist (ndarray): The histogram. Shape (n_clusters)
        """
        # Input - list of descriptors for a single image
        # Use the fitted kmeans model to label the descriptors
        # Make a histogram of the labels using np.histogram. Remember to specify the amount of bins (n_clusters) and the range (0, n_clusters - 1)
        # return the histogram
        
        # self.kmeans.predict(descriptors, )

    def predict(self, img):
        """
        Finds the closest match in the training set to the given image

        Parameters
        ----------
        img (ndarray): The query image. Shape (height, width [,3])

        Returns
        -------
        match_idx (int): The index of the training image there was closest, -1 if there was non descriptors in the image
        """
        # Compute descriptors for the image
        # Calculate the histogram using hist()
        # Calculate the distance to each histogram in the database
        # Return the index of the histogram in the database with the smallest distance
        
        # Extract features and find labels
        _, des = self.extractor.detectAndCompute(img, None)
        labels = self.kmeans.predict(des)
        
        # Compute and normalize histogram
        hist, _ = np.histogram(labels, self.n_clusters, (self.min_hist,  self.max_hist), density=False)
        hist = np.float64(hist) / self.n_features
        hist = hist.reshape((1, -1))
        
        dist = np.sum(self.idf * (self.histograms - hist) ** 2, axis=1)

        return np.argmin(dist)
        

def split_data(dataset, test_size=0.1):
    """
    Loads the images and split it into a train and test set

    Parameters
    ----------
    dataset (str): The path to the dataset
    test_size (float): Represent the proportion of the dataset to include in the test split

    Returns
    -------
    train_img (list): The images in the training set. Shape (n_images)
    test_img (list): The images in the test set. Shape (n_images)
    """
    # Load the images and split it into a train and test set using train_test_split from sklearn
    
    img_names = os.listdir(dataset)
    n = len(img_names)
    images = []
    for img_name in img_names:
        img_path = os.path.join(dataset, img_name)
        images.append(cv.imread(img_path))
    
    np.random.shuffle(images)
    train_img = []
    test_img = []
    for i in range(n):
        if i < test_size * n:
            test_img.append(images[i])
        else:
            train_img.append(images[i])
            
    return train_img, test_img


def make_stackimage(query_image, match_image=None):
    """
    hstack the query and match image

    Parameters
    ----------
    query_image (ndarray): The query image. Shape (height, width [,3])
    match_image (ndarray): The match image. Shape (height, width [,3])

    Returns
    -------
    stack_image (ndarray): The stack image. Shape (height, 2*width [,3])
    """
    match_found = True
    if match_image is None:
        match_image = np.zeros_like(query_image)
        match_found = False

    if len(query_image.shape) != len(match_image.shape):
        if len(query_image.shape) != 3:
            query_image = cv.cvtColor(cv.COLOR_GRAY2BGR, query_image)
        if len(match_image.shape) != 3:
            match_image = cv.cvtColor(cv.COLOR_GRAY2BGR, match_image)

    height1, width1, *_ = query_image.shape
    height2, width2, *_ = match_image.shape
    height = max([height1, height2])
    width = max([width1, width2])

    if len(query_image.shape) == 2:
        stack_shape = (height, width * 2)
    else:
        stack_shape = (height, width * 2, 3)

    if match_found:
        put_text(query_image, "top_center", "Query")
        put_text(match_image, "top_center", "Match")

    stack_image = np.zeros(stack_shape, dtype=match_image.dtype)
    stack_image[0:height1, 0:width1] = query_image
    stack_image[0:height2, width:width + width2] = match_image

    if not match_found:
        put_text(stack_image, "top_center", "No features found")

    return stack_image


if __name__ == "__main__":
    
    abs_dir = os.path.dirname( os.path.abspath(__file__))
    dataset = os.path.join(abs_dir, '../data/COIL20/images')
    # dataset = os.path.join(abs_dir, '../data/StanfordDogs/images')
    
    n_clusters = 20  # number of cluster. Type int
    n_features = 100  # number of features. Type int
    assert n_clusters != 0 and n_features != 0, "Remember to change n_clusters and n_features in main"

    # Split the data
    train_img, test_img = split_data(dataset)

    # Make the BoW and train it on the training data
    bow = BoW(n_clusters, n_features)
    bow.train(train_img)

    win_name = "query | match"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 1024, 600)

    # Find matches to every test image
    for i, img in enumerate(test_img):
        # Find the closest match in the training set
        idx = bow.predict(img)
        if idx != -1:
            # If a match was found make a show_image with the query and match image
            show_image = make_stackimage(img, train_img[idx])
        else:
            # If a match was not found make a show_image with the query image
            print("No features found")
            show_image = make_stackimage(img)

        # Show the result
        put_text(show_image, "bottom_center", f"Press any key.. ({i}/{len(test_img)}). ESC to stop")
        cv.imshow(win_name, show_image)
        key = cv.waitKey()
        if key == 27:
            break
