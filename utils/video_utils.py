import cv2

def read_video(video_path):
    cap=cv2.VideoCapture(video_path)# Video Capture Object
    frames=[]
    while True:
        ret, frame=cap.read()
        if not ret: # flag to check if the video has ended
            break
        frames.append(frame) # Append the frame to the list of frames
    return frames

def save_video(output_video_frames, output_video_path):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Output Video Codec
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0])) # output(width,height)
    for frame in output_video_frames:
        out.write(frame)
    out.release()