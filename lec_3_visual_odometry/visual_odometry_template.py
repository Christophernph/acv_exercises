import sys
import os
import numpy as np
import cv2
import logging

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):    
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
        # Scale reference to determine size of translations
        self.scale_ref = np.linalg.norm(self.gt_poses[0][:3, 3] - self.gt_poses[1][:3, 3])

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
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
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detects and computes keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        
        
        # Determine if last three images can be extracted
        valid = i > 1
        
        # Extract images
        img0 = self.images[i - 2] if valid else None
        img1 = self.images[i - 1]
        img2 = self.images[i]
        
        # Find keypoints and descriptors
        if valid:
            kp0, des0 = self.orb.detectAndCompute(img0, None)
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
    
        # Use Flann matcher (query, train)
        matches_12 = self.flann.knnMatch(des1, des2, k=2)
        if valid:
            matches_01 = self.flann.knnMatch(des0, des1, k=2)
            matches_02 = self.flann.knnMatch(des0, des2, k=2)
        
        common_matches = []
        if valid:   
            # Find features common between all three images
            for match_01 in matches_01:
                for match_02 in matches_02:
                    
                    # Verify img0 feature is the same
                    if match_01[0].queryIdx != match_02[0].queryIdx:
                        continue
                    
                    for match_12 in matches_12:
                        
                        if match_01[0].trainIdx != match_12[0].queryIdx:
                            continue
                        
                        if match_12[0].trainIdx == match_02[0].trainIdx:
                            common_matches.append([match_12, match_01, match_02])
        else:
            common_matches = matches_12
            
        q0 = []
        q1 = []
        q2 = []
        thresh = 0.5
        for match_tuple in common_matches:
            if valid:
                m12 = match_tuple[0]
                m01 = match_tuple[1]
                m02 = match_tuple[2]
                
                if m12[0].distance < thresh * m12[1].distance and m01[0].distance < thresh * m01[1].distance and m02[0].distance < thresh * m02[1].distance:
                    q0.append(kp0[m01[0].queryIdx].pt)
                    q1.append(kp1[m12[0].queryIdx].pt)
                    q2.append(kp2[m12[0].trainIdx].pt)
            else:
                if match_tuple[0].distance < thresh*match_tuple[1].distance:
                    q1.append(kp1[match_tuple[0].queryIdx].pt)
                    q2.append(kp2[match_tuple[0].trainIdx].pt)
        

        q0 = np.array(q0)
        q1 = np.array(q1)
        q2 = np.array(q2)
        
        return q0, q1, q2
        

    def get_pose(self, q0, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T
        
        # print(q1.shape, q2.shape)
        E, mask  = cv2.findEssentialMat(q1, q2, self.K)    # Could specify RANSAC parameters but nah
        
        right_pair = self.decomp_essential_mat(E, q1, q2)
        
        T = self._form_transf(right_pair[0], right_pair[1])
        
        scale = self.determine_scale(q0, q1, q2, T)
        
        # Save for next iteration and fix scale
        self.last_transform = T
        self.last_transform[:3, 3] *= scale
        
        return self.last_transform
    
    def determine_scale(self, q0, q1, q2, T):
        
        # If first iteration, return ground truth scale
        if len(q0) == 0:
            return self.scale_ref
        
        # Construct projection matrices
        P0 = self.P @ np.linalg.inv(self.last_transform)
        P1 = self.P @ np.eye(4, 4)
        P2 = self.P @ T
        
        # Triangulate points
        Q_01 = cv2.triangulatePoints(P0, P1, q0.T, q1.T)
        Q_12 = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        
        # Normalize
        Q_01 = Q_01 / Q_01[3, :]
        Q_12 = Q_12 / Q_12[3, :]
        
        # Find average relative scale (could be performed for even more point pairs)
        r = []
        for i in range(Q_01.shape[1] - 1):
            n = np.linalg.norm(Q_01[:, i] - Q_01[:, i + 1])
            d = np.linalg.norm(Q_12[:, i] - Q_12[:, i + 1])
            if d != 0:
                r.append(n / d)

        print(np.median(r), np.mean(r))
        return np.median(r)

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z
        
        # Decompose into the two possible rotations and translations
        R1, R2, t = cv2.decomposeEssentialMat(E)
        combinations = [(R1, t.reshape(-1)), (R1, -t.reshape(-1)), (R2, t.reshape(-1)), (R2, -t.reshape(-1))]
        best_idx = 0
        best_value = -1
        for idx, combo in enumerate(combinations):
            
            # Form possible transformation
            T = self._form_transf(combo[0], combo[1])
            
            # Form the two projection matrices
            P1 = self.P @ np.eye(4, 4)
            P2 = self.P @ T
            
            # Triangulate points (as given in first camera)
            Q = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
            Q = Q / Q[3, :]
            
            # Determine if points are in front of cameras
            z = np.array([0, 0, 1]).reshape((1, 3))
            z_valid1 = (z @ np.hstack((np.eye(3, 3), np.zeros((3, 1)))) @ Q) > 0
            z_valid2 = (z @ T[:3, :] @ Q) > 0
            z_valid = np.hstack((z_valid1, z_valid2))
            # z_valid = np.logical_and(z_valid1, z_valid2)
            
            # If most number so far in front of cameras, save this one
            if np.sum(z_valid) > best_value:
                best_value = np.sum(z_valid)
                best_idx = idx

        return [combinations[best_idx][0], combinations[best_idx][1]]
        

def main():
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = abs_dir + '/../data/KITTI_sequence_1'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        
        if i == 0:
            cur_pose = gt_pose
        else:
            q0, q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q0, q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
