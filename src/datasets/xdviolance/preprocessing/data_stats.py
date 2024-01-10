import os
import moviepy.editor as mp

def extract_statistics(video_path,folder):
    if (folder):

        for file in os.listdir(video_path):
            b = str.split(file,".")
            if b[1] == "DS_Store":
                continue
            else:

                # Load the video file
                video = mp.VideoFileClip(video_path+file)

                # Extract the audio sample rate
                audio_sample_rate = video.audio.fps

                # Extract the video FPS
                video_fps = video.fps

                # Extract the video length
                video_length = video.duration

                # Extract the video resolution
                video_resolution = video.size

                # Print the extracted statistics
                # print(f"\nvideo path: {video_path+file}")
                # print(f"Audio sample rate: {audio_sample_rate}")
                # print(f"Video FPS: {video_fps}")
                # print(f"Video length: {video_length}")
                # print(f"Video resolution: {video_resolution}")
    else:
        # Load the video file
        video = mp.VideoFileClip(video_path)

        # Extract the audio sample rate
        audio_sample_rate = video.audio.fps

        # Extract the video FPS
        video_fps = video.fps

        # Extract the video length
        video_length = video.duration

        # Extract the video resolution
        video_resolution = video.size

        # Print the extracted statistics
        # print(f"\nvideo path: {video_path}")
        # print(f"Audio sample rate: {audio_sample_rate}")
        # print(f"Video FPS: {video_fps}")
        # print(f"Video length: {video_length}")
        # print(f"Video resolution: {video_resolution}")
    return audio_sample_rate,video_fps,video_length,video_resolution
# Example usage
# video_path = "/Users/michelepresutto/Desktop/Intership Folder/data_script/data_set/train/"
# sr,fps,vid_len,resolution = extract_statistics(video_path,folder=True)
