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


PATH = "/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/"
def onehot(label):
    #a = label.split("-")
    a = label
    onehot_a = []
    if(a[0] == "G"):
        onehot_a = [0,0,0,0,0,1,0]
        #onehot_a = 5
    if(a[0] == "B1"):
        onehot_a = [1,0,0,0,0,0,0]
        #onehot_a = 0

    if(a[0] == "B2"):
        onehot_a = [0,1,0,0,0,0,0]
        #onehot_a = 1

    if(a[0] == "B4"):
        onehot_a = [0,0,1,0,0,0,0]
        #onehot_a = 2
    if(a[0] == "B5"):
        onehot_a = [0,0,0,1,0,0,0]
        #onehot_a = 3
    if(a[0] == "B6"):
        onehot_a = [0,0,0,0,1,0,0]
        #onehot_a = 4
    if(a[0] == "A"):
        onehot_a = [0,0,0,0,0,0,1]
        #onehot_a = 6

    return onehot_a
def get_max_size(path):
    max_size = (0,0)
    for i in os.listdir(path):
        for j in os.listdir(os.path.join(path,i)):

            #print("\npath:\n",j)
            img = Image.open(os.path.join(path,i,j,"frame0001.png"))
            #print(type(img))
            img_sz = img.size
            #print(img_sz)
            break
        if(max_size < img_sz):
            max_size = img_sz
    return max_size
def pad_to_max(img,max_size):
    im_sz=img.size
    pad_x = max_size[0]-im_sz[0]
    pad_y = max_size[1]-im_sz[1]
    transform = T.Pad((0,0,pad_x, pad_y))
    img = transform(img)
    return img

class XDViolence(Dataset):
    def __init__(self, root_dir,n_sample, transform=None):
        self.root_dir = root_dir
        #self.n_frames = n_frames
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
        smpl_files = sorted(os.listdir(os.path.join(self.root_dir)))
        frame_files = sorted(os.listdir(os.path.join(self.root_dir, video_folder)))
        #print(smpl_files)
        #n_frames = min(self.n_frames, len(frame_files))
        sample = []
        
        max_size = get_max_size(PATH)
        #print("\nMAX SIZE : ",max_size,"\n\n")
        labe = []
        fram_ = []
        labe_ = []
        spec_ = []
        #if self.n_sample < len(frame_files)
        #for j in range(len(smpl_files)):
        frames = []
        labels = []
        spectr = []
        for i in range(self.n_sample):
            if frame_files[i] != "full_spectrogram.png" and frame_files[i] != "output.mp3":

                frame_path = os.path.join(self.root_dir, video_folder, frame_files[i])
                #print("\nframe path\n\n",frame_path)
                base, ext = os.path.splitext(video_folder)
                #print("extention name:",ext)
                filename_frame = os.listdir(frame_path)
                bas,ext_ = os.path.splitext(frame_path)
                bo ,label = bas.split("label_")
                label,__ = label.split(".")
                label = label.split("-")
                label_ = onehot(label) 
                spect = Image.open(os.path.join(frame_path,"c_spectrogram.png"))
                spect = spect.convert('L')
                frame = Image.open(os.path.join(frame_path,"frame0001.png"))
                spect = pad_to_max(spect,max_size)
                frame = pad_to_max(frame,max_size)

                spect = torch.tensor(np.array(spect))
                frame = torch.tensor(np.array(frame))

                frames.append(frame)
                spectr.append(spect)
                labels.append(torch.tensor(np.argmax(label_)))
        
            fram = torch.stack(frames)
            spec = torch.stack(spectr)            
            labe = torch.stack(labels)
        # fram_.append(torch.tensor(fram))
        # spec_.append(torch.tensor(spec))
        # labe_.append(torch.tensor(labe))
        # fram_ = torch.stack(fram_)
        # spec_ = torch.stack(spec_)            
        # labe_ = torch.stack(label_)
            #print("labeeeee\n",labe)
        return fram,spec,labe#,self.label
def compute_weight_class(loader):
    
    print("\necco train data_label\n",loader.label)
    for i in loader.label:
        print("\nlabels\n",i)


    

#     print("\n..weight computing..\n")
#     print("total number of samples",len(data.label))
#     print("number of labels from a random samples", data.label[1])
def load_data(path):
    data = XDViolence(path, n_sample = 5)
    data_loader = DataLoader((data), batch_size=3, shuffle=True)

    return data_loader
# train_data = XDViolence('/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/', n_frames=10,n_sample=20)
# #print("awalla\n", train_data[1])
# video_dataloader = DataLoader((train_data), batch_size=1, shuffle=True)
# print("len dataloader\n",len(video_dataloader))
# i, (f) = next(enumerate(video_dataloader))
# # print("lennnnn\n",len(f))
# print("len of first element of batch\n",(f[1][0][0]))


# val_data = XDViolence('/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/test/', n_frames=2)

# i, (frames,spect,label) = next(enumerate(video_dataloader))
# spect_np = spect[0][0][0].numpy()
# spect = Image.fromarray(spect_np)
# print(type(spect))
# print("\nframes len\n",len(frames))

#visualize the output 

#frames_ = f[1][10][0].numpy()
#print("label:\n",(f[2][30][0].numpy()))
# print(len(frames_))
#frames_ = Image.fromarray(frames_)
#frames_.save("vediamochee.png")
# dataload = load_data("/raid/home/dvl/mpresutto/vol/DynMM/FusionDynMM/datasets/xdviolence/train/")
# print(dataload.dataset)
#print("\ndataloader size\n",len(dataload))

#for i, (frame,spect,label) in (enumerate(dataload)):
    #print(i,"labello\n\n")
    # print(label)
    #print("\n",frame.shape,"\n",spect.shape,"\n",label.shape,"\n")
    #for sample in frame:
        #print(sample.size())
    #     print("sample size\n",i,"iter \n",(sample.size()))
    # for sp in spect:

    #     print("spect size \n",i,(sp.size()))
    #for label_ in label:
        #print("labels \n",(label_))
        
        
        

