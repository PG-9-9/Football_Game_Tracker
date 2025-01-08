import numpy as np
import cv2

class ViewTransformer:
    """
    A class to transform points from a trapezoidal (camera) view to a rectangular (top-down) view.
    """

    def __init__(self):
        """
        Initialize the ViewTransformer with default court dimensions and vertices.
        """
        court_width = 68  # Width of the court in meters
        court_length = 23.32  # Length of the court in meters

        # Trapezoidal view of the court (camera perspective)
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom left
            [265, 275],   # Top left
            [910, 260],   # Top right
            [1640, 915]   # Bottom right
        ])

        # Rectangular view of the court (top-down perspective)
        self.target_vertices = np.array([
            [0, court_width],        # Bottom left
            [0, 0],                  # Top left
            [court_length, 0],       # Top right
            [court_length, court_width]  # Bottom right
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute the perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        """
        Transform a point from the trapezoidal view to the rectangular view.

        Args:
            point (numpy.ndarray): Point to transform.

        Returns:
            numpy.ndarray or None: Transformed point or None if outside the court.
        """
        # Check if the point is inside the trapezoidal court
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Reshape the point for transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer)

        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        """
        Add transformed positions to object tracks.

        Args:
            tracks (dict): Dictionary of object tracks.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
