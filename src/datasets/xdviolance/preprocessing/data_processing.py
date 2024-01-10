import os
import subprocess
import time
from spectrogram_2 import extract
from data_stats import extract_statistics
import cv2
from PIL import Image

def extract_spect(video_path, output_dir, fps=None):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Extract audio
    audio_command = ["ffmpeg", "-i", video_path, output_dir+"/output.mp3"]
    subprocess.run(audio_command)
    #compute the spectrogram
    extract(output_dir)
def extract_single_frame(video_path,time_sec,frame_indx,output_dir,use_seconds = False):
    #the frame can be extracted using time in seconds or using the frame index depending on the param use_seconds
    os.makedirs(output_dir, exist_ok=True)
    ss = time.strftime('%H:%M:%S', time.gmtime(float(time_sec)))

    if use_seconds:
        frame_command = ['ffmpeg', '-ss',ss,"-i", video_path, "-frames:v",str(1)]
        frame_command += [f'{output_dir}/frame%04d.png']
    else:
        frame_command = ['ffmpeg', '-i', video_path, '-vf', f"select=eq(n\,{frame_indx})", '-vframes', '1']
        frame_command += [f'{output_dir}/frame%04d.png']
    
    subprocess.run(frame_command)

def cut_spectro(path,frame_indx,total_frames):
    im = Image.open(path)
    width, height = im.size
    #print("\nwidth of the spectrogram\n",width)
    #bins_per_frame = width/total_frames
    #print("\nbins per frame\n",bins_per_frame)
    center = frame_indx#width/2

    # Setting the points for cropping the image
    left = center-320
    top = 0
    right = center+320
    bottom = height
    
    im1 = im.crop((left, top, right, bottom))    
    #im1.show()
    return im1
#/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/videos
#dir = "/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/videos/"
#dir_out = "/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/"
dir = "/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/subset_test/"
dir_out = "/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/test/"

for file in os.listdir("/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/subset_test"):
    sr,fps,vid_len,resolution = extract_statistics(dir+file, folder=False)
    cap = cv2.VideoCapture(dir+file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_spect(dir+file, dir_out+file)
    for frm in range(100,300,10):
        
    #print(f'Total number of frames: {total_frames}')
        extract_single_frame(dir+file,0,int(round(frm)),dir_out+file+"/frame_"+str(frm),use_seconds = False)
    #extract_window_from_spect(dir+file,vid_len/2,2)
        cutted_spect = cut_spectro(dir_out+file+"/full_spectrogram.png",frm,total_frames)
        print("\n\nsize of the cutted spectrogram\n\n",(cutted_spect.size))
        cutted_spect.save(dir_out+file+"/frame_"+str(frm)+"/c_spectrogram.png")
