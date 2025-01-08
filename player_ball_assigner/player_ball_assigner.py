import sys

# Add the parent directory to the path to import utility functions
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    """
    A class to assign a ball to the closest player based on bounding box positions.
    """

    def __init__(self):
        """
        Initialize the PlayerBallAssigner with default parameters.
        """
        self.max_player_ball_distance = 70  # Maximum allowable distance to assign the ball to a player

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign the ball to the nearest player within a maximum distance.

        Args:
            players (dict): Dictionary of players with their bounding boxes.
            ball_bbox (tuple): Bounding box of the ball.

        Returns:
            int: ID of the player closest to the ball, or -1 if no player is within range.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Measure distance from the ball to the player's left and right bottom corners
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)

            # Select the minimum distance for assignment
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
