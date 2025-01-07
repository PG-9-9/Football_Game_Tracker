from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Stores the colors of the two teams
        self.player_team_dict = {}  # Maps player ID to their assigned team

    def get_clustering_model(self, image):
        # Reshape the image into a 2D array (pixels x RGB channels)
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters (team colors)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the region of interest (ROI) corresponding to the bounding box
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Use only the top half of the bounding box image for clustering
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half of the image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel in the top half image
        labels = kmeans.labels_

        # Reshape the labels back to the shape of the top half image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify the player's cluster by examining the clusters of corner pixels
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)  # Most common corner cluster
        player_cluster = 1 - non_player_cluster  # Opposite cluster is assumed to be the player cluster

        # Get the RGB color of the player's cluster center
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []  # List to store detected player colors

        # Iterate through each player's detection
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]  # Get the bounding box for the player
            player_color = self.get_player_color(frame, bbox)  # Extract the player's dominant color
            player_colors.append(player_color)

        # Perform K-means clustering on the collected player colors to determine team colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans  # Save the K-means model for later use

        # Assign the two team colors from the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]  # Team 1 color
        self.team_colors[2] = kmeans.cluster_centers_[1]  # Team 2 color

    def get_player_team(self, frame, player_bbox, player_id):
        # Check if the player's team is already determined
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract the player's dominant color from the frame
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team based on the player's color using the K-means model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Adjust team ID to be 1 or 2

        # Special case: Assign player with ID 91 to team 1
        if player_id == 91:
            team_id = 1

        # Store the player's team in the dictionary
        self.player_team_dict[player_id] = team_id

        return team_id
