import cv2
import pickle
import numpy as np
import sys
import os

# Add the parent directory to the path to import utility functions
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    """
    A class to estimate and manage camera movement between frames in a video sequence.
    """

    def __init__(self, frame):
        """
        Initialize the CameraMovementEstimator with a frame.

        Args:
            frame (numpy.ndarray): The first frame of the video sequence.
        """
        self.minimum_distance = 5  # Minimum distance to consider a point as a feature

        # Parameters for the Lucas-Kanade optical flow algorithm
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Convert the initial frame to grayscale
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask to exclude regions from feature detection
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Mask the first 20 pixels from the left
        mask_features[:, 900:1050] = 1  # Mask specific columns

        # Parameters for feature detection
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjust the object positions in the tracks based on the camera movement.

        Args:
            tracks (dict): Dictionary containing object tracks.
            camera_movement_per_frame (list): List of camera movements for each frame.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculate the camera movement between frames.

        Args:
            frames (list): List of video frames.
            read_from_stub (bool): Whether to read camera movement data from a stub file.
            stub_path (str): Path to the stub file.

        Returns:
            list: List of camera movement vectors for each frame.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        # Process the first frame
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()
                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw camera movement information on the frames.

        Args:
            frames (list): List of video frames.
            camera_movement_per_frame (list): List of camera movement vectors for each frame.

        Returns:
            list: List of annotated frames.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames
