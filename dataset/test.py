from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

from PIL import Image
import numpy as np
import os
import pickle
import torch
import pandas as pd

def concatenate(datas,labels):
    con_data=datas[0]
    con_label=labels[0]
    for i in range(1,len(datas)):
        con_data=np.concatenate((con_data,datas[i]),axis=0)
        con_label=np.concatenate((con_label,labels[i]),axis=0)
    return con_data,con_label
def get_test_set(test_set_name, data_manager, test_transform, test_set_path):
    print(test_set_name)
    test_set = {
        "imagenet" : ImagenetTestDataset,
        "imagenet100" : ImagenetTestDataset,
        "imagenet1000" : ImagenetTestDataset,
        "tiny_imagenet" : TinyImagenetTestDataset,
        "cifar100" : Cifar100TestDataset,
        "mini_imagenet" : MiniImagenetTestDataset,
        "cifar10" : Cifar10TestDataset,
        "urbansound8k" : UrbanSound8KTestDataset,
        "dailynsports" : DailynSportsTestDataset,
        "audioset" : AudioSetTestDataset
    }
    if test_set == "imagenet100":
        return ImagenetTestDataset(test_set_path=test_set_path, data_manager=data_manager, test_transform=test_transform, num_class=100)
    else:
        if test_set_path is not None:
            return test_set[test_set_name](test_set_path=test_set_path, data_manager=data_manager, test_transform=test_transform)
        else:
            return test_set[test_set_name](data_manager=data_manager, test_transform=test_transform)


class UrbanSound8KTestDataset(Dataset):
    """
    Defines UrbanSound8K as for the others pytorch datasets.
    """
    # Replace /data with relative path data
    def __init__(self, test_set_path='data', train: bool=False, 
                 data_manager=None, test_transform=None,
                 download: bool=False) -> None:
        self.root = test_set_path
        self.data_path = os.path.dirname(os.path.dirname(self.root))   
        self.train = train
        self.download = download
        self.data_manager = data_manager

        # Load data directly, now it is just the filename
        self.data, self.targets= self.load_data(self.root)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        self.transform = test_transform
        self.TestData = []
        self.TestLabels = []
    def concatenate(self,datas,labels):
        con_data = np.stack(datas,axis=0)
        con_label=np.array(labels)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            fnames = self.data[self.targets == actual_label]
            for path in fnames: 
                # load img 
                path = self.data_path+'/'+path
                with open(path,'rb') as f:
                        file = Image.open(f)
                        img = file.convert('RGB')
                        data = self.transform(img)
                        
                datas.append(data)
                labels.append(label)

        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
    def load_data(self,fpath):
        file_list = [idx_file for idx_file in os.scandir(fpath) if idx_file.name == 'test.txt']
        fpath = file_list[0].path
        data = []
        labels = []
        for x in range (10):
            data.append([])
            labels.append([])
        lines = open(fpath)
        # path_prefix = '/data/project/rw/'
        for line in lines:
            arr = line.strip().split()
            label = int(arr[1])
            # when a full set is not used
            if(label>=len(labels)):
                break
            data[label].append(arr[0])
            labels[label].append(label)
        return data, labels
    def __getitem__(self, index):
        vec, target = self.TestData[index], self.TestLabels[index]
        return vec,target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[self.targets==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])


class DailynSportsTestDataset(Dataset):
    """
    Defines DailynSports as for the others pytorch datasets.
    """
    # Replace /data with relative path data
    def __init__(self, test_set_path='data', train: bool=False, 
                 data_manager=None, test_transform: transforms=None,
                 download: bool=False) -> None:
        root = test_set_path
        self.train = train
        self.download = download
        self.data_manager = data_manager

        self.data = []
        self.targets = []
        
        for a_num in range(1,20): # 19 categories
            for p_num in range(1,9): # 8 users 
                for s_num in range(49,61):#range_txt: # 1-48 train, 49-60 test
                    np_data = np.loadtxt(f"data/dailynsports_data/a"+str(a_num).zfill(2)+"/p"+str(p_num).zfill(1)+"/s"+str(s_num).zfill(2)+".txt",dtype=np.float32,delimiter=",")
                    self.data.append(np_data) # dimension = (125,45)
                    self.targets.append(a_num)
                    
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        self.TestData = []
        self.TestLabels = []
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]
        if task_id in self.data_manager.classes_per_task: # Key error
            for label in self.data_manager.classes_per_task[task_id]:
                
                if label in self.TestLabels:
                    continue

                actual_label = self.data_manager.map_int_label_to_str_label[label]

                data = self.data[self.targets == actual_label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
        else: print("KeyError: " + str(task_id))
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
        
    def __getitem__(self, index):
        vec, target = self.TestData[index], self.TestLabels[index]
        return vec,target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[self.targets==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])

ALL_LETTERS = "abcdefghijklmnopqrstuvwxyz0123456789\n !\"&'(),-.:;>?[]}"
NUM_LETTERS = len(ALL_LETTERS) # 54


class ImagenetTestDataset(Dataset):
    def __init__(self,
                 test_set_path='/data',
                 data_manager=None,
                 split='val',
                 test_transform=None,
                 target_transform=None,
                 num_class=1000
                 ):
        self.test_set_path = test_set_path
        self.data_manager = data_manager
        self.test_transform = test_transform

        self.num_class = num_class

        if self.num_class == 1000:
            self.data_paths, self.labels = self.load_data('data/imagenet-1000/val.txt')
        elif self.num_class == 100:
            self.data_paths, self.labels = self.load_data('data/imagenet-100/val.txt')

        self.data = list()
        self.targets = list()

    def load_data(self, fpath):
        data = []
        labels = []

        lines = open(fpath)
        
        for i in range(self.num_class):
            data.append([])
            labels.append([])

        for line in lines:
            arr = line.strip().split()
            data[int(arr[1])].append(arr[0])
            labels[int(arr[1])].append(int(arr[1]))

        return data, labels

    def append_task_dataset(self, task_id):
        for label in self.data_manager.classes_per_task[task_id]:
            actual_label = self.data_manager.map_int_label_to_str_label[label]

            if label in self.targets:
                continue
            for data_path in self.data_paths[actual_label]:
                data_path = os.path.join(self.test_set_path, data_path) # Hard-coded path
                with open(data_path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                self.data.append(img)
                self.targets.append(label)
    def clean_task_dataset(self):
        if self.num_class == 1000:
            self.data_paths, self.labels = self.load_data('data/imagenet-1000/val.txt')
        elif self.num_class == 100:
            self.data_paths, self.labels = self.load_data('data/imagenet-100/val.txt')

        del self.data[:]
        del self.targets[:]
        self.data = list()
        self.targets = list()
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class TinyImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, test_set_path='data', root: str='data', train: bool=False, 
                 data_manager=None, test_transform: transforms=None,
                 download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = test_set_path
        self.train = train
        self.download = download
        self.data_manager = data_manager
        self.test_transform = test_transform

        if download:
            from google_drive_downloader import GoogleDriveDownloader as gdd
            # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

        self.TestData = []
        self.TestLabels = []
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)

    def __getitem__(self, index):
        img, target = Image.fromarray(np.uint8(255 *self.TestData[index])), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])

class MiniImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self,root='/data',
                 test_set_path='./data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True) -> None:
        self.root = '/data'
        self.train = train
        self.data_manager = data_manager
        self.test_transform = test_transform
        
        self.data = []
        self.targets = []

        self.TestData = []
        self.TestLabels = []

        train_in = open(root+"/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open(root+"/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open(root+"/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))

        TEST_SPLIT = 1 / 6

        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            x_test = cur_x[: int(600 * TEST_SPLIT)]
            y_test = cur_y[: int(600 * TEST_SPLIT)]
            test_data.append(x_test)
            test_label.append(y_test)

        self.data = np.concatenate(test_data)
        self.targets = np.concatenate(test_label)
        self.targets = torch.from_numpy(self.targets).type(torch.LongTensor)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])

class Cifar100TestDataset(CIFAR100):
    def __init__(self,root='./data',
                 test_set_path='./data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets,dtype=int) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])
        del self.TestData, self.TestLabels
        self.TestData,self.TestLabels=[],[]


class Cifar10TestDataset(CIFAR10):
    def __init__(self,root='./data',
                 test_set_path='./data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform
        print("test_transform : ", self.test_transform)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        print("data_manager.classes_per_task[task_id] : ", self.data_manager.classes_per_task[task_id])
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])
        
class AudioSetTestDataset(Dataset):

    # Replace /data with relative path data
    def __init__(self, test_set_path='data', train: bool=False, 
                 data_manager=None, test_transform=None,
                 download: bool=False) -> None:
        root = test_set_path
        self.test_idx = f'{root}/test_idxs.txt'
        self.data_path = f'{root}/spectrogram/'
        self.data_manager = data_manager

        from pandas import read_csv
        indices = read_csv(self.test_idx, header=None, dtype={0:str,1:'int32'})
        self.data = {x:list(indices[indices[1]==x][0]) for x in indices[1].unique()}
        self.targets = {x:[x]*len(self.data[x]) for x in self.data}
        self.transform = test_transform
        self.TestData = []
        self.TestLabels = []
    def load_data(self, label):
        label_paths = self.data[label]
        datas = []
        for path in label_paths: 
            path = f'{self.data_path}/{path}'
            data = np.load(path)
            if data.shape[0] <1001:
                data = np.pad(data, ((0,1001-data.shape[0]),(0,0)))
            datas.append(data)
                
        return datas

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]
            data=self.load_data(actual_label)
            datas.append(data)
            
            labels.append(np.array(self.targets[label]))
        if len(datas)>0 and len(labels)>0:
            datas,labels=concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            # FIXME 
            if self.TestData.shape[0] != self.TestLabels.shape[0]:
                if  self.TestData.shape[0] < self.TestLabels.shape[0]:
                    self.TestLabels = self.TestLabels[:self.TestData.shape[0]]
                else: 
                    self.TestData = self.TestData[:self.TestLabels.shape[0]]

    def __getitem__(self, index):
        vec, target = self.TestData[index], self.TestLabels[index]
        return vec,target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[self.targets==label]
    def clean_task_dataset(self):

        np.delete(self.data,np.s_[:])
        np.delete(self.targets,np.s_[:])