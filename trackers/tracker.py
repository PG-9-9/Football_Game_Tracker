from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../') # Add the parent directory to the path to import the bbox utils
from utils import get_center_of_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd


class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def interpolate_ball_positions(self,ball_positions):
        ball_positions=[x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate the missing values
        df_ball_positions=df_ball_positions.interpolate()
        df_ball_positions=df_ball_positions.bfill()# Backward fill the remaining NaN values , especially the first few frames

        ball_positions=[{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):

        batch_size=20 # Number of frames to process at a time, to prevent memory overflow
        detections=[]
        for i in range(0,len(frames),batch_size):
            
            # overide the goalkeeper class with the player class, use predict and then track.
            detections_batch=self.model.predict(frames[i:i+batch_size], conf=0.1) # Minimum Confidence threshold is set to 0.1
            detections+=detections_batch

        return detections

    def get_object_tracks(self, frames,read_from_stub=False, stub_path=None):


        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks=pickle.load(f)
                return tracks

        detections=self.detect_frames(frames)

        tracks={
            "players":[], # {frame_num: {track_id: {"bbox": [x1, y1, x2, y2]}, {track_id: {"bbox": [x1, y1, x2, y2]}}, ...} 
            "referees":[],
            "ball":[],	
        }

        for frame_num, detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k, v in cls_names.items()}

            # Convert the detections into the supervision format
            detections_supervision=sv.Detections.from_ultralytics(detection)
            
            # Convert the goal keeper class to player class
            for obj_ind, class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id]=='goalkeeper':
                    detections_supervision.class_id[obj_ind]=cls_names_inv['player']

            # Track the objects
            # Adding the detections to the tracker
            detection_with_track=self.tracker.update_with_detections(detections_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_track:
                bbox=frame_detection[0].tolist()# 0 Detection, 1 Mask, 2 Confidence
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id]={"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id]={"bbox":bbox}

            for frame_detection in detections_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1]={"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        
        y2 = int(bbox[3]) # bottom y

        x_center, _= get_center_of_bbox(bbox)

        width = get_bbox_width(bbox)

        cv2.ellipse(frame, 
                    center=(x_center, y2),
                    axes=(int(width),int(0.35*width)), # Ellipse has both minor and major axis
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )
        
        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center-rectangle_width//2 # Top left x
        x2_rect=x_center+rectangle_width//2 # Bottom right x
        y1_rect=(y2-rectangle_height//2)-15 # Top left y
        y2_rect=(y2+rectangle_height//2)-15 # Bottom right y

        if track_id is not None:
                    cv2.rectangle(frame,
                                (int(x1_rect),int(y1_rect)),
                                (int(x2_rect),int(y2_rect)),
                                color,
                                cv2.FILLED)
                    
                    x1_text = x1_rect+12 # X position of the text + 12 in padding
                    if track_id > 99:# If the track id is greater than 99, then move the text to the left
                        x1_text -=10 # Move the text to the left
                    
                    cv2.putText(
                        frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2
                    )

        return frame
    
    def draw_traingle(self,frame,bbox,color):
        y=int(bbox[1]) # Top y, because we need to draw the triangle at the top of the bbox
        x,_=get_center_of_bbox(bbox)

        triangle_points=np.array([ # Inverted Triangle
            [x,y], # Peak(Bottom) of the triangle (Inverted), so it will be downwards
            [x-10,y+20], # Top left
            [x+10,y+20] # Top right
        ])
        # Draw the triangle
        cv2.drawContours(frame,
                        [triangle_points],
                        0, # Index of the contour to draw
                        color, # Color of the contour
                        cv2.FILLED)#-1 to fill the contour
        # Draw the traingle border
        cv2.drawContours(frame,
                        [triangle_points],
                        0, # Index of the contour to draw
                        (0,0,0), # Color of the contour
                        2)#2 for the thickness of the contour

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame=frame.copy()
            player_dict=tracks["players"][frame_num]
            refree_dict=tracks["referees"][frame_num]
            ball_dict=tracks["ball"][frame_num]

            # Draw the ellipse for the players
            for track_id, player in player_dict.items():
                color=player.get("team_color", (0,0,255)) # Default color is red
                frame = self.draw_ellipse(frame, player["bbox"], color ,track_id)

            # Draw the ellipse for the referees
            for _, referee in refree_dict.items(): #No need to track id the referees
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255),)

            # Draw the triangle for the ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))
            
            output_video_frames.append(frame)
        
        return output_video_frames