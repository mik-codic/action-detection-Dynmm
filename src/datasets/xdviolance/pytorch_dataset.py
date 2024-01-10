import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

def onehot(label):
    a = label.split("-")
    onehot_a = []
    if(a[0] == "G"):
        onehot_a = [0,0,0,0,0,1,0]
        #onehot_a = 0
    if(a[0] == "B1"):
        onehot_a = [1,0,0,0,0,0,0]
        #onehot_a = 1

    if(a[0] == "B2"):
        onehot_a = [0,1,0,0,0,0,0]
        #onehot_a = 2

    if(a[0] == "B4"):
        onehot_a = [0,0,1,0,0,0,0]
        #onehot_a = 4
    if(a[0] == "B5"):
        onehot_a = [0,0,0,1,0,0,0]
        #onehot_a = 5
    if(a[0] == "B6"):
        onehot_a = [0,0,0,0,1,0,0]
        #onehot_a = 6
    if(a[0] == "A"):
        onehot_a = [0,0,0,0,0,0,1]
        #onehot_a = 3

    return onehot_a
def get_max_size(path):
    for i in os.listdir(path):
        img = Image.open(path+"/frame0001.png")
        img_sz = img.size()
        max_size = img_sz

def ImgToSpect(wavw):

    # Compute spectrogram
    frequencies, times, spectrogram = scipy.signal.spectrogram(wavw, 44100)

    # Plot spectrogram
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of mono audio')
    plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    plt.close()  # Close the plot to free up memory

    return pil_image

class XDViolence(Dataset):
    def __init__(self, root_dir, n_frames,n_sample, transform=None):
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.transform = transform or ToTensor()
        self.video_folders = sorted(os.listdir(root_dir))
        self.frames = []
        self.spect = []
        self.label = []
        self.n_sample = n_sample
    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):     
        video_folder = self.video_folders[idx]
        frame_files = sorted(os.listdir(os.path.join(self.root_dir, video_folder)))
        n_frames = min(self.n_frames, len(frame_files))

        
        sample = []
       
        frames = []
        labels = []
        spectr = []
        for i in range(n_frames):
            frame_path = os.path.join(self.root_dir, video_folder, frame_files[i])
            base, ext = os.path.splitext(frame_path)
            bas,ext_ = os.path.splitext(video_folder)
            bo ,label = bas.split("label_")
            print("\nlabels example\n",label)
            label = onehot(label[0])
            print("\nlabels\n",label)
            #label = ord(label[0])
            print("\n",self.root_dir+"/"+video_folder+"/spectrogram.png"+"\n")
            spect = Image.open(self.root_dir+"/"+video_folder+"/spectrogram.png")
            spect = spect.convert('L')
            #spect = spect.convert('RGB')
            if ext.lower() == '.png':
                f = Image.open(frame_path)
                print("size of image\n",f.size)
                spect = spect.resize(f.size)
            if spect == None:
                print("\nerror: spectrogram not found\n")
            spect = torch.tensor(np.array(spect))
            if ext.lower() == '.wav' or ext.lower() == ".DS_Store":
                continue
            elif ext.lower() == '.png':
                
                frame = Image.open(frame_path)
                print("wtf\n",frame.size)
                frame = torch.tensor(np.array(frame))
                label = torch.tensor(np.array(label))
                frames.append(frame)
                spectr.append(spect)
                labels.append(label)
                self.frames.append(frames)
                self.label.append(labels)
                self.spect.append(spectr)
        return self.frames,self.spect,self.label
def compute_weight_class(loader):
    
    print("\necco train data_label\n",loader.label)
    for i in loader.label:
        print("\nlabels\n",i)


    

#     print("\n..weight computing..\n")
#     print("total number of samples",len(data.label))
#     print("number of labels from a random samples", data.label[1])
def load_data(path):
    data = XDViolence(path, n_frames=10,n_sample = 1)
    data_loader = DataLoader((data) ,batch_size=2, shuffle=True)
    return data_loader
#train_data = XDViolence('/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/', n_frames=10,n_sample=1)

# video_dataloader = DataLoader((train_data), batch_size=2, shuffle=True)
# print(len(video_dataloader))
# i, (f) = next(enumerate(video_dataloader))
# print("lennnnn\n",len(f))
# print("len of train data xd\n",len(train_data[0]))


# val_data = XDViolence('/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/test/', n_frames=2)

# i, (frames,spect,label) = next(enumerate(video_dataloader))
# spect_np = spect[0][0][0].numpy()
# spect = Image.fromarray(spect_np)
# print(type(spect))
# print("\nframes len\n",len(frames))

# frames_ = frames[1][0][0].numpy()
# print(len(frames_))
# frames_ = Image.fromarray(frames_)
# #frames_.show()

# spect.show()


