import cv2
import pickle
import numpy as np
import sys
import os

sys.path.append('../') # Add the parent directory to the path to import the bbox utils
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):

        self.minimum_distance= 5 # Minimum distance to consider a point as a feature

        # Parameters for the Lucas-Kanade optical flow
        self.lk_params=dict(
            winSize=(15,15), # Size of the search window at each pyramid level
            maxLevel=2, # 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # Termination criteria of the iterative search algorithm (after the specified maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon)
        )
        
        first_frame_grayscale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask_features=np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20]=1# mask the first 20 pixels from the top
        mask_features[:,900:1050]=1# mask the last 150 pixels from the top

        self.features=dict(
            maxCorners=100, # Maximum number of corners to return
            qualityLevel=0.3, # The minimum accepted quality of image corners
            minDistance=3, # Minimum possible Euclidean distance between the returned corners
            blockSize=7,# Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
            mask=mask_features
        )


        pass

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position=track_info['position']
                    camera_movement=camera_movement_per_frame[frame_num]
                    position_adjusted=(position[0]-camera_movement[0],position[1]-camera_movement[1]) # Adjust the position
                    tracks[object_type][frame_num][track_id]['position_adjusted']=position_adjusted


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub if it exists
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        camera_movement=[[0,0]]*len(frames)# [dx, dy]

        old_gray=cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)# Get the first frame
        old_features=cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1,len(frames)):
            frame_gray=cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features,_,_=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance=0
            camera_movement_x,camera_movement_y=0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point=new.ravel()
                old_features_point=old.ravel()

                distance=measure_distance(new_features_point,old_features_point)

                if distance>max_distance:
                    max_distance=distance
                    camera_movement_x,camera_movement_y=measure_xy_distance(old_features_point, new_features_point)
            
            if max_distance>self.minimum_distance:
                camera_movement[frame_num]=[camera_movement_x,camera_movement_y] 
                old_features=cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray=frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        
        return camera_movement
    

    def draw_camera_movement(self,frames, camera_movement_per_frame):

        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame=frame.copy()
            overlay=frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha=0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement,y_movement=camera_movement_per_frame[frame_num]
            frame=cv2.putText(frame,f"Camera Movement X: {x_movement:2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
            frame=cv2.putText(frame,f"Camera Movement Y: {y_movement:2f}",(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames



