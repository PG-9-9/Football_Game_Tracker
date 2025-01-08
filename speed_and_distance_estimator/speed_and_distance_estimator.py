import sys
sys.path.append('../')  # Add the parent directory to the path to import the bbox utils
from utils import measure_distance, get_foot_position
import cv2

class SpeedAndDistanceEstimator:
    """
    A class to estimate and annotate speed and distance traveled by tracked objects in a video.
    """

    def __init__(self):
        """
        Initialize the SpeedAndDistanceEstimator with default parameters.
        """
        self.frame_window = 5  # Number of frames to consider for speed estimation
        self.frame_rate = 24  # Frame rate of the video

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Compute and add speed and distance information to the object tracks.

        Args:
            tracks (dict): Dictionary containing object tracks across frames.
        """
        total_distance_covered = {}

        for object_type, object_tracks in tracks.items():
            if object_type in ["ball", "referees"]:
                continue  # Skip ball and referee objects

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_mps = distance_covered / time_elapsed
                    speed_kmph = speed_mps * 3.6

                    total_distance_covered.setdefault(object_type, {}).setdefault(track_id, 0)
                    total_distance_covered[object_type][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in object_tracks[frame_num_batch]:
                            continue
                        tracks[object_type][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[object_type][frame_num_batch][track_id]['distance'] = total_distance_covered[object_type][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        """
        Annotate speed and distance information on video frames.

        Args:
            frames (list): List of video frames.
            tracks (dict): Dictionary containing object tracks.

        Returns:
            list: List of annotated video frames.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type in ["ball", "referees"]:
                    continue

                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed')
                        distance = track_info.get('distance')

                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        text_position = (int(position[0]), int(position[1] + 40))

                        cv2.putText(frame, f"Speed: {speed:.2f} km/h", text_position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Distance: {distance:.2f} m",
                                    (text_position[0], text_position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            output_frames.append(frame)

        return output_frames
