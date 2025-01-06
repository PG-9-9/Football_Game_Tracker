from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read the video
    video_frames=read_video('input_videos/08fd33_4.mp4')

    # Initialize the tracker
    tracker=Tracker('models/best.pt')

    tracks=tracker.get_object_track(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    

    # Draw output
    ## Draw the bounding boxes
    output_video_frames=tracker.draw_annotations(video_frames,tracks)


    # Save the video
    save_video(output_video_frames, 'output_videos/output_video.avi')

  

if __name__ == "__main__":
    main()