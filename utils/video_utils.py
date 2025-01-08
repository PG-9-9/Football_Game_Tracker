import cv2

def read_video(video_path):
    """
    Read a video file and extract its frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: List of frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)  # Video capture object
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # Check if the video has ended
            break
        frames.append(frame)
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Save a list of frames as a video file.

    Args:
        output_video_frames (list): List of frames to save.
        output_video_path (str): Path to the output video file.
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Output video codec
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,  # Frame rate
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0])  # Frame dimensions
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
