import os

import cv2
import face_recognition as fr
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from lib.visualization.image import choose_face


class EigenFaces:
    def __init__(self, unprocessed_imgs, labels, face_size, n_vecs):
        """
        Parameters
        ----------
        unprocessed_imgs (ndarray): The training images. Shape (n, height, width)
        labels (list): The labels (names) to the faces in the images. Shape (n)
        face_size (tuple): The image size the faces should be in before face recognition
        """
        self.face_size = face_size
        face_cascade_file = f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)

        # Find faces in the unprocessed_imgs
        x_train, self.y_train = self.process_images(unprocessed_imgs, labels)

        # Train the EigenFaces and project the x_train data
        self.x_train = self.train(x_train, n_vecs=n_vecs)

    def process_images(self, unprocessed_imgs, labels):
        """
        Processes the images.

        Parameters
        ----------
        unprocessed_imgs (list): List with images. Shape (n_images)
        labels (list): List with the names of the people's faces there should be in the images. Shape (n_images)

        Returns
        -------
        x (ndarray): The faces from the images as vectors. Shape (n_faces,self.face_size[0]*self.face_size[1])
        y (ndarray): The names for the faces. Shape (n_faces)
        """
        # Prepare the images for the EigenFaces algorithm:
        # For each image, convert it to grayscale and detect faces using the CascadeClassifier
        # If multiple faces detected remove the face not corresponding to the label (e.g. use cv2 to show both images and select the correct one)
        # Crop the face using the detected bounding box, resize to self.face_size, and flatten
        # Save crop and label in a list each, x and y
        # If no face is detected discard label
        # Return x, y
        for idx, img in enumerate(unprocessed_imgs):
            unprocessed_imgs[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            unprocessed_imgs[idx] = cv2.equalizeHist(unprocessed_imgs[idx])
            
            #-- Detect faces
            faces = self.face_cascade.detectMultiScale(unprocessed_imgs[idx])
            
            for (x,y,w,h) in faces:
                center = (x + w // 2, y + h // 2)
                faceROI = unprocessed_imgs[idx][y:y+h,x:x+w]
                cv2.imshow('Capture - Face detection', faceROI)
            
        pass

    def train(self, x, n_vecs):
        """
        Calculats the mean, eigenfaces and resturns projected face vectors

        Parameters
        ----------
        x (ndarray): Face image vectors. Shape (n_faces, self.face_size[0]*self.face_size[1])
        n_vecs (int): The number of the most significant eigenfaces to use

        Returns
        -------
        x_projected (ndarray) The projected face vectors. Shape (n_faces, n_vecs)
        """
        # Calculate the mean value for each pixel, i.e. along axis 0
        # Calculate the covariance of x (output should be of size (n_pixels x n_pixels) and not (n_images x n_images))
        # Use Eigen value decomposition on the covariance matrix
        # Remove all except the n_vecs best Eigen vectors
        # Project the input to face space and return them
        pass

    def project(self, face_vector):
        """
        Project the face vector into face space

        Parameters
        ----------
        face_vector (ndarray): The face image vector. Shape (face_size[0]*face_size[1])

        Returns
        -------
        face_project (ndarray): The projection of the face vectors. Shape (n_vecs)
        """
        # Project the image to face space
        pass

    def predict(self, face_vector):
        """
        Predict how's face are present in the face vector

        Parameters
        ----------
        face_vector (ndarray): The face image vector. Shape (face_size[0]*face_size[1])

        Returns
        -------
        label (str): The name of the predicted person
        """
        # Use self.project(...) to project the image to face space
        # Calculate the distance to each encoding in the train set
        # Return the label of the training example closest to the new example


# Optional
class FaceNet():
    def __init__(self, x_train, y_train):
        """
        Parameters
        ----------
        x_train (ndaray): The training images
        y_train (list): The training labels
        """
        pass

    def process_images(self, images, labels):
        """
        Processes the images.

        Parameters
        ----------
        images (list) List with images
        labels (list): List with the names of the people's faces there should be in the images

        Returns
        -------
        x (ndarray): The face encodings
        y (ndarray): The names for the faces

        """
        pass

    def predict(self, encoding):
        """
        Predict how's face are present in the face vector

        Parameters
        ----------
        encoding (ndarray): Face encoding

        Returns
        -------
        label (str): The name of the predicted person
        """
        pass


def split_data(dataset, test_size=0.2):
    """
    Loads the images and split it into a train and test set

    Parameters
    ----------
    dataset (str): The path to the dataset
    test_size (float): Represent the proportion of the dataset to include in the test split

    Returns
    -------
    x_train (list): The images in the training set. Shape (n_train, height, width)
    x_test (list): The images in the test set. Shape (n_test, height, width)
    y_train (list): The labels in the training set. Shape (n_train)
    y_test (list): The labels in the test set. Shape (n_test)
    """
    images, labels = [], []
    for label in os.listdir(dataset):
        path = os.path.join(dataset, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                images.append(cv2.imread(os.path.join(os.path.join(dataset, label, file))))
                labels.append(label)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    dataset = '../data/5-celebrity-faces'
    n_vecs = 25  # number of the most significant eigenfaces to use. Type int
    face_size = (256, 256)  # size of the face to use
    assert n_vecs != 0 and face_size != (0,0), "Remember to change n_vecs and face_size in main"


    # Load the dataset and split it into train and test set
    x_train, x_test, y_train, y_test = split_data(dataset)

    # -- EigenFaces --
    # Create a EigenFaces detector with the training images
    eigenface = EigenFaces(x_train, y_train, face_size=face_size, n_vecs=n_vecs)

    # Process the test images
    x_test_proc, y_test_proc = eigenface.process_images(x_test, y_test)
    # Preform face recognition
    y_pred = [eigenface.predict(image) for image in x_test_proc]
    # Print the classification report
    print(classification_report(y_test_proc, y_pred))

    # -- FaceNet --
    # Create a FaceNet detector with the training images
    facenet = FaceNet(x_train, y_train)
    # Process the test images
    x_test_proc, y_test_proc = facenet.process_images(x_test, y_test)
    # Preform face recognition
    y_pred = [facenet.predict(encoding) for encoding in x_test_proc]
    # Print the classification report
    print(classification_report(y_test_proc, y_pred))
