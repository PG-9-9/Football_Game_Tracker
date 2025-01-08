from sklearn.cluster import KMeans

class TeamAssigner:
    """
    A class to assign teams to players based on dominant colors extracted from player regions.
    """

    def __init__(self):
        """
        Initialize the TeamAssigner with empty team colors and player-to-team mappings.
        """
        self.team_colors = {}  # Dictionary to store the colors of the two teams
        self.player_team_dict = {}  # Maps player IDs to their assigned team

    def get_clustering_model(self, image):
        """
        Perform K-means clustering on the provided image.

        Args:
            image (numpy.ndarray): Image to cluster.

        Returns:
            sklearn.cluster.KMeans: Fitted K-means model.
        """
        image_2d = image.reshape(-1, 3)  # Reshape image into a 2D array (pixels x RGB channels)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color of a player based on their bounding box.

        Args:
            frame (numpy.ndarray): Current video frame.
            bbox (tuple): Bounding box of the player.

        Returns:
            numpy.ndarray: Dominant color of the player.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0] / 2), :]
        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_.reshape(top_half_image.shape[:2])
        corner_clusters = [
            labels[0, 0],
            labels[0, -1],
            labels[-1, 0],
            labels[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors based on player detections.

        Args:
            frame (numpy.ndarray): Current video frame.
            player_detections (dict): Detected player regions with bounding boxes.
        """
        player_colors = [self.get_player_color(frame, det["bbox"]) for det in player_detections.values()]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine the team of a player based on their color.

        Args:
            frame (numpy.ndarray): Current video frame.
            player_bbox (tuple): Bounding box of the player.
            player_id (int): ID of the player.

        Returns:
            int: Assigned team ID (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
