from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict
import pickle
from array import array 
import time
import gc
import random
from _utils.sampling import multi_task_sample_update_to_RB


class ReplayImageNet1k(Dataset):
    num_file_for_label = dict()

    def __init__(self, rb_path, rb_size, transform, sampling, agent,dataset='imagenet1000',device='cpu',postfix=None):
        self.rb_path = rb_path
        self.transform = transform
        self.rb_size = rb_size
        self.filled = 0
        self.sampling = sampling
        self.agent = agent
        self.device = device

        self.data = list()
        self.targets = list()
        self.filenames = list()
        self.tracker = list()

        self.offset = dict()
        self.len_per_cls = dict()

        if self.sampling == "reservoir_der":
            self.logits = list()
    
        self.get_vec_time = []
        self.get_target_time = []
        self.aug_time = []
        self.total_time = []

    def get_sub_data(self, label):
        if label in self.offset and label in self.len_per_cls:
            st = self.offset[label]
            en = self.offset[label]+self.len_per_cls[label]
            sub_data = self.data[st:en]
            sub_label = self.targets[st:en]
            sub_filename = self.filenames[st:en]
            sub_index = list(range(st,en))
            
            return sub_data, sub_label,sub_filename,sub_index
        else:
            sub_data = []
            sub_index = []
            sub_label = []
            for idx in range(len(self.data)):
                if self.targets[idx] == label:
                    sub_data.append(self.data[idx])
                    sub_label.append(label)
                    sub_index.append(idx)
                    sub_filename.append(self.filenames[idx])
            return sub_data, sub_label, sub_filename,sub_index

    def __len__(self):
        assert len(self.data) == len(self.filenames)
        num_cls = len(self.len_per_cls)

        if num_cls>0: 
            return min(len(self.data),self.rb_size,self.len_per_cls[num_cls-1]+self.offset[num_cls-1] )
        else: 
            return min(len(self.data),self.rb_size)
    def is_filled(self):
        if len(self.data) == 0:
            return False
        else:
            return True

    def __getitem__(self, idx):
        if self.agent in ["der","derpp","derpp_distill", "tiny","aser"]:
            return self.getitem_online(idx)

        else:
            return self.getitem_offline(idx)
    def getitem_offline(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        target = self.targets[idx]
        filename = self.filenames[idx]
        

        return idx, data, target, filename
    def getitem_online(self, idx):
        #print(self.targets)
        #print("IDX : ", idx)

        img = self.data[idx]

        
        img = self.transform(img)
        label = self.targets[idx]
        data_id = self.tracker[idx]
        
        if self.agent in ["der","derpp_distill","derpp"]:
            logit = self.logits[idx]
            logit = torch.as_tensor(array('f', logit))


            return idx, img, label, logit, data_id

        else:
            return idx, img, label, data_id

    def resize(self, new_size, samples_seen, swap_manager,fetch_func, test_set,task_id, delete=False,old_len_per_cls=None, old_offset=None, old_rb_size=None,max_rb=None): 
        print('BEFORE RESIZE')
        print(self.len_per_cls)
        print(self.offset)
        print(self.rb_size)
        print(len(self))
        
        if new_size == self.rb_size: 
            print('No need to resize')
            return 3 
        exit_code = -1 
        print(f'replay_dataset resize: {self.rb_size} -> {new_size}, delete = {delete}')
        
        new_mem_per_cls = int(new_size/len(self.len_per_cls))
        print(new_mem_per_cls)
        
        ## only deletion
        if new_size <= len(self.data): 
            self.rb_size = new_size
            if delete==True: 
                del self.data[self.rb_size:]
                del self.targets[self.rb_size:]
                del self.filenames[self.rb_size:]
        self.rb_size = new_size
        
        if delete == False: 
            old_len_per_cls, old_offset, old_rb_size=None, None, None
        self.fetch_from_storage(samples_seen, swap_manager, fetch_func,test_set,task_id,old_len_per_cls, old_offset, old_rb_size)

        print('AFTER RESIZE')
        # print(self.filenames)
        # print(self.targets)
        print(self.len_per_cls)
        print(self.offset)
        print(self.rb_size)
        print(len(self))
    def fetch_from_storage(self, samples_seen, swap_manager, fetch_func,test_set,task_id,old_len_per_cls=None, old_offset=None, old_rb_size=None):
        num_cls = len(self.offset)
        memory_per_cls = min(int(self.rb_size / num_cls ), int(samples_seen / len(self.offset)))
        fetch_targets = []
        fetch_files = []
        self.len_per_cls = {x:memory_per_cls for x in range(num_cls)}
        self.offset = {x:x*memory_per_cls for x in range(num_cls)}
        print('FETCH FROM STORAGE', flush=True)
        for i in self.offset: 
            print(f'FETCH label {i}', flush=True)
            if swap_manager.saver: 
                n_samples = min(memory_per_cls,swap_manager.saver.get_num_file_for_label_for_swap(i))
                target_list = [i]*n_samples
                self.len_per_cls[i]=n_samples
                label_list= [f'{i}_{count+1}'  for count in range(min(memory_per_cls,swap_manager.saver.get_num_file_for_label_for_swap(i))) ]
            else: 
                target_list = [i]*memory_per_cls
                label_list= [f'{i}_{count+1}'  for count in range(memory_per_cls)]
            for k in range(i, len(self.offset)): 
                if i==0: break
                self.offset[k] = self.offset[k-1] + self.len_per_cls[k-1]
            fetch_files.extend(label_list)
            fetch_targets.extend(target_list)
            

        
        fetch_idx = [i for i in range(len(fetch_targets))]
        for j in range(len(fetch_targets)):
            if j < len(self.targets):  self.targets[j] = fetch_targets[j]
            else: 
                self.targets.append(fetch_targets[j])
        fetch_func(self, fetch_idx,fetch_targets,task_id, filenames=fetch_files, testset=test_set)
 