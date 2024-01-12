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
        self.frames_per_fold = []
        cumulative_num_frames = 0
        for i in self.video_folders:
            num_frames = sorted(os.listdir(os.path.join(self.root_dir,i)))
            cumulative_num_frames = cumulative_num_frames+len(num_frames)-2
            self.frames_per_fold.append(cumulative_num_frames)
    def __len__(self):
        total_num_frames = 0
        for i in self.video_folders:
            num_frames = sorted(os.listdir(os.path.join(self.root_dir,i)))
            total_num_frames = total_num_frames+(len(num_frames)-2)
        #print(total_num_frames)
        return total_num_frames

    def __getitem__(self, idx):     
        print("\nindex",idx)
        print(self.frames_per_fold)
        for i in range(0,len(self.frames_per_fold)):
            if idx < self.frames_per_fold[i]:
                video_folder = self.video_folders[i]
                if i != 0:
                    index_in_fold = idx-self.frames_per_fold[i-1]
                    break
                else: 
                    index_in_fold = idx
                    break
        max_size = get_max_size(self.root_dir) 
        print("index_in_fold",index_in_fold)

        # video_folder = self.video_folders[idx]
        frame_files = sorted(os.listdir(os.path.join(self.root_dir, video_folder)))
        frame_files.remove("full_spectrogram.png")
        frame_files.remove("output.mp3")
        #print("frame files",frame_files)

        image = Image.open(os.path.join(self.root_dir,video_folder,frame_files[index_in_fold],"frame0001.png"))

        spect = Image.open(os.path.join(self.root_dir,video_folder,frame_files[index_in_fold],"c_spectrogram.png"))
        spect = spect.convert('L')
        spect = spect.resize(image.size)
        spect = pad_to_max(spect,max_size)
        image = pad_to_max(image,max_size)
        bas,ext_ = os.path.splitext(video_folder)
        bo ,label = bas.split("label_")
        #print("\nlabels example\n",label)
        label = onehot(label[0])

        image = torch.tensor(np.array(image))
        spect = torch.tensor(np.array(spect))
        label = torch.tensor(np.array(label))
        #label = torch.tensor([torch.argmax(lab) for lab in label])
        return image,spect,label
       
def compute_weight_class(loader):
    
    print("\necco train data_label\n",loader.label)
    for i in loader.label:
        print("\nlabels\n",i)
    
def load_data(path):
    data = XDViolence(path, n_frames=0,n_sample = 1)
    data_loader = DataLoader((data) ,batch_size=10, shuffle=True)
    return data_loader

def pad_to_max(img,max_size):
    im_sz=img.size
    pad_x = max_size[0]-im_sz[0]
    pad_y = max_size[1]-im_sz[1]
    transform = T.Pad((0,0,pad_x, pad_y))
    img = transform(img)
    return img


def get_max_size(path):
    max_size = (0,0)
    for i in os.listdir(path):
        for j in os.listdir(os.path.join(path,i)):

            img = Image.open(os.path.join(path,i,j,"frame0001.png"))
            #print(type(img))
            img_sz = img.size
            #print(img_sz)
            break
        if(max_size < img_sz):
            max_size = img_sz
    return max_size

val_data = load_data("/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train")
print("len data",len(val_data))
for i, (frames,spect,label) in enumerate(val_data):
    label = [np.argmax(lab) for lab in label]
    print(label)
