import os
import numpy as np
import cv2
from scipy.optimize import least_squares

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images_l = self._load_images(os.path.join(data_dir, 'image_l'))
        self.images_r = self._load_images(os.path.join(data_dir, 'image_r'))

        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        self.disparities = [
            np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        self.fastFeatures = cv2.FastFeatureDetector_create()
        # self.fastFeatures.setNonmaxSuppression(0)    # this

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """
        # Split the image into tiles and detect the 10 best keypoints in each tile
        # Return a 1-D list of all keypoints
        # Hint: use sorted(keypoints, key=lambda x: -x.response)
        
        rows = img.shape[0] // tile_h
        cols = img.shape[1] // tile_w
        
        keypoints = []
        for row in range(rows):
            for col in range(cols):
                tile = img[row * tile_h: (row + 1) * tile_h, col * tile_w: (col + 1) * tile_w]
                kp = self.fastFeatures.detect(tile, None)
                kp = np.array(sorted(kp, key=lambda x: -x.response))
                
                for idx in range(len(kp)):
                    if idx == 10:
                        break
                    
                    kp[idx].pt = (kp[idx].pt[0] + col * tile_w, kp[idx].pt[1] + row * tile_h)   # (x, y)
                    keypoints.append(kp[idx])
        
        return keypoints

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints using cv2.KeyPoint_convert
        # Use cv2.calcOpticalFlowPyrLK to estimate the keypoint locations in the second frame. self.lk_params contains parameters for this function.
        # Remove all points which are not trackable, has error over max_error, or where the points moved out of the frame of the second image
        # Return a list of the original converted keypoints (after removal), and their tracked counterparts
        
        
        pts1 = cv2.KeyPoint_convert(kp1)
        pts2, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, **self.lk_params)
        
        # Mask based on status, error, and out of frame
        mask = np.logical_and(_st, _err < max_error).reshape(-1)
        mask = np.logical_and(mask, pts2[:, 0] > 0) # x less than zero
        mask = np.logical_and(mask, pts2[:, 1] > 0) # y less than zero
        mask = np.logical_and(mask, pts2[:, 0] < img2.shape[1]) # x more than width
        mask = np.logical_and(mask, pts2[:, 1] < img2.shape[0]) # y more than height
        pts1 = pts1[mask]
        pts2 = pts2[mask]
        
        # image_stack = np.hstack((img1, img2))
        # plt.imshow(image_stack)
        # plt.scatter(pts1[:10, 0], pts1[:10, 1])
        # plt.scatter(pts2[:10, 0] + pts2.shape[1], pts2[:10, 1])
        # plt.show()
        
        return pts1, pts2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        # Get the disparity for each keypoint
        # Remove all keypoints where disparity is out of bounds
        # calculate keypoint location in right image by subtracting disparity from x coordinates
        # return left and right keypoints for both frames
        
        # Cast to ints
        q1_l = q1.astype(np.int)
        q2_l = q2.astype(np.int)
        
        # Mask q1 within bounds 
        mask = disp1[q1_l[:, 1], q1_l[:, 0]] > min_disp
        mask = np.logical_and(mask, disp1[q1_l[:, 1], q1_l[:, 0]] < max_disp)
        
        # Mask q2 within bounds
        mask = np.logical_and(mask, disp2[q2_l[:, 1], q2_l[:, 0]] > min_disp)
        mask = np.logical_and(mask, disp2[q2_l[:, 1], q2_l[:, 0]] < max_disp)
        
        # Apply mask
        q1_l = q1_l[mask]
        q2_l = q2_l[mask]
        
        # Find features in right image by subtracting disparity
        q1_r = q1_l - disp1[q1_l[:, 1], q1_l[:, 0]]
        q2_r = q2_l - disp2[q2_l[:, 1], q2_l[:, 0]]
        
        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images

        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        
        # Triangulate points
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        
        # Normalize
        Q1 = Q1[:3, :] / Q1[3, :]
        Q2 = Q2[:3, :] / Q2[3, :]
        
        return Q1.T, Q2.T
        

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Implement RANSAC to estimate the pose using least squares optimization
        # Sample 6 random point sets from q1, q2, Q1, Q2
        # Minimize the given residual using least squares to find the optimal transform between the sampled points
        # Calculate the reprojection error when using the optimal transform with the whole point set
        # Redo with different sampled points, saving the transform with the lowest error, until max_iter or early termination criteria met
        
        all_indices = np.arange(q1.shape[0])
        for i in range(max_iter):
            
            # Sample points
            np.random.shuffle(all_indices)
            indices = all_indices[:6]
            
                    

    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get the tiled keypoints
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparitie
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix


def main():
    
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = abs_dir + '/../data/KITTI_sequence_1'  # Try KITTI_sequence_2 too
    
    vo = VisualOdometry(data_dir)

    #play_trip(vo.images_l, vo.images_r)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i < 1:
            cur_pose = gt_pose
        else:
            transf = vo.get_pose(i)
            cur_pose = np.matmul(cur_pose, transf)
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Stereo Visual Odometry",
                             file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
