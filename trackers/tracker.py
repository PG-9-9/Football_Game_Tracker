from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd

# Add the parent directory to the path to import the bbox utils
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    """
    A class to detect, track, and annotate objects (players, referees, and ball) in video frames.
    """

    def __init__(self, model_path):
        """
        Initialize the Tracker with a YOLO model and a ByteTrack tracker.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        """
        Add the calculated position (center or foot position) to object tracks.

        Args:
            tracks (dict): Dictionary of object tracks.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions in frames.

        Args:
            ball_positions (list): List of ball positions (bounding boxes).

        Returns:
            list: Interpolated ball positions.
        """
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Perform interpolation and backward fill
        df_ball_positions = df_ball_positions.interpolate().bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        """
        Detect objects in video frames using the YOLO model.

        Args:
            frames (list): List of video frames.

        Returns:
            list: Detection results for each frame.
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track objects across video frames.

        Args:
            frames (list): List of video frames.
            read_from_stub (bool): Whether to read tracks from a pre-saved stub.
            stub_path (str): Path to the stub file.

        Returns:
            dict: Dictionary of tracked objects.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detections_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper class to player class
            for obj_ind, class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detections_supervision.class_id[obj_ind] = cls_names_inv['player']

            detection_with_track = self.tracker.update_with_detections(detections_supervision)

            # Initialize empty dictionaries for each frame
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_track:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse around the object.

        Args:
            frame (numpy.ndarray): Video frame.
            bbox (list): Bounding box of the object.
            color (tuple): Color of the ellipse.
            track_id (int, optional): Object's track ID.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(x_center, y2), axes=(int(width), int(0.35 * width)),
                    angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2)

        if track_id:
            # Draw rectangle for ID
            cv2.putText(frame, f"{track_id}", (x_center - 10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle above the object.

        Args:
            frame (numpy.ndarray): Video frame.
            bbox (list): Bounding box of the object.
            color (tuple): Color of the triangle.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw team ball control statistics on the frame.

        Args:
            frame (numpy.ndarray): Video frame.
            frame_num (int): Current frame number.
            team_ball_control (list): List of ball control statistics.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350, 850), (1900, 970),(255,255,255) ,-1)
        alpha=0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1] # Get the team ball control till the current frame
        
        # Count the number of frames each team has the ball
        team_1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0] # Returns the number of frames where team 1 has the ball
        team_2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0] # Returns the number of frames where team 2 has the ball

        team_1=team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2=team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame,f"Team 1 Ball Control: {team_1:.2f}",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2:.2f}",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Annotate the video frames with object information.

        Args:
            video_frames (list): List of video frames.
            tracks (dict): Dictionary of object tracks.
            team_ball_control (list): List of ball control statistics.

        Returns:
            list: Annotated video frames.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in tracks["referees"][frame_num].items():
                self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in tracks["ball"][frame_num].items():
                self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames
