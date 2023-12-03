from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
from collections import deque
from lib import utils
from multiprocessing import Manager
from array import array
import random

def multi_task_sample_update_to_RB(replay_dataset, stream_dataset, *arg):
    if replay_dataset.sampling == 'ringbuffer_imagenet':
        return ringbuffer_imagenet(replay_dataset, stream_dataset,*arg)
    return ringbuffer_offline(replay_dataset, stream_dataset, *arg)

def ringbuffer_offline(replay_dataset, stream_dataset, val=False, mem_per_cls=None):
    if val == True:
        eviction_idx_list = []
    num_new_label = 0
    if stream_dataset:
        for label in stream_dataset.classes_in_dataset:
            if label not in replay_dataset.offset or label not in replay_dataset.len_per_cls:
                num_new_label += 1
    if not mem_per_cls:
        num_classes = (len(replay_dataset.offset) + num_new_label)
        memory_per_cls = int(replay_dataset.rb_size / num_classes)
    else: 
        num_classes = len(replay_dataset.offset) 
        memory_per_cls = mem_per_cls
    print(f"SAMPLING EM, memory per cls = ", memory_per_cls)
    # temporarily lift length limit 
    if val == False: replay_dataset.filled[0] =replay_dataset.rb_size
    old_offset, old_len_per_cls = replay_dataset.offset, replay_dataset.len_per_cls
    replay_dataset.len_per_cls = {x:memory_per_cls for x in range(num_classes)}
    replay_dataset.offset = {x:x*memory_per_cls for x in range(num_classes)}
    # skip 
    if old_offset == replay_dataset.offset and old_len_per_cls == replay_dataset.len_per_cls and stream_dataset==None: 
        if val == False: replay_dataset.filled[0] = sum(replay_dataset.len_per_cls.values())
        print(f'SKIP SAMPLING')
        return
    old_targets = [i for i in replay_dataset.targets]
    old_filenames = [i for i in replay_dataset.filenames]
    for i in range(num_classes): 
        sub_stream_data,sub_stream_label,sub_stream_filenames,sub_stream_idx = [],[],[],[]
        if stream_dataset:
            sub_stream_data,sub_stream_label,sub_stream_filenames,sub_stream_idx = stream_dataset.get_sub_data(i)
            total = (len(sub_stream_idx) + old_len_per_cls[i]) if i in old_len_per_cls else len(sub_stream_idx) 
            if memory_per_cls >= total: 
                cls_offset = total
            else: cls_offset=memory_per_cls
        else: cls_offset = memory_per_cls 
        if i < len(replay_dataset.offset)-1: 
            replay_dataset.offset[i+1] = replay_dataset.offset[i] + cls_offset
        replay_dataset.len_per_cls[i] = cls_offset      
        for j in range(cls_offset):
            st = replay_dataset.offset[i]
            # Old data triming       
            if i < (len(old_offset)) and j < old_len_per_cls[i]:
                # moving old data 
                old_st = old_offset[i]
                old_len = old_len_per_cls[i]
                new_len = replay_dataset.len_per_cls[i]
                if new_len < old_len:
                    old_idx = old_st+old_len-new_len+j
                else: 
                    old_idx = old_st + j
                old_data_shm = replay_dataset.data_shm[old_idx]
                old_data = np.ndarray(replay_dataset.vec_shape,replay_dataset.vec_dtype,buffer=old_data_shm.buf)
                new_data_shm = replay_dataset.data_shm[st+j]
                new_data = np.ndarray(replay_dataset.vec_shape,replay_dataset.vec_dtype,buffer=new_data_shm.buf)
                new_data[:] = old_data[:]
                # print(old_idx)
                replay_dataset.targets[st+j] = old_targets[old_idx]
                replay_dataset.filenames[st+j] = old_filenames[old_idx]
                # print(f'replay[{st+j}] <- replay[{old_idx}]')
                # df.write(f'replay[{st+j}] <- replay[{old_idx}] = {old_filenames[old_idx]}\n')
                # print(old_targets[old_idx],old_filenames[old_idx])
            # Insert new samples from new classes
            elif stream_dataset: 
                if i not in old_offset or i not in old_len_per_cls: 
                    stream_idx = total-cls_offset+j
                else: 
                    stream_idx = total-cls_offset+j - old_len_per_cls[i]
                # print(f'2 replay[{st+j}] <- stream[{stream_idx}]',end=': ')
                # print(sub_stream_label[stream_idx],sub_stream_filenames[stream_idx])
                
                replay_dataset[st+j] = sub_stream_data[stream_idx],sub_stream_label[stream_idx],sub_stream_filenames[stream_idx]
                # _, data,label,filename = stream_datset[sub_stream_idx[total-memory_per_cls+j]]
                # replay_dataset[st+j] = data,label,filename
                # print(f'replay == stream {(data==b).all()}')

                if val == True: 
                    eviction_idx_list.extend(sub_stream_idx[-cls_offset:])
        # for i in range(len(replay_dataset)):
        #     dfile.write(f'{replay_dataset.filenames[i]}, ')    # print(replay_dataset.filenames)
        # print(f'sampling output: {replay_dataset.targets}')
        if val == False: 
            replay_dataset.filled[0] = sum(replay_dataset.len_per_cls.values())
            # print(f'REPLAY FILLED: {replay_dataset.filled[0]}')
def ringbuffer_imagenet(replay_dataset, stream_dataset, val=False, mem_per_cls=None):

    print('BEFORE SAMPLING')
    # print(replay_dataset.filenames)
    # print(replay_dataset.targets)
    print(replay_dataset.len_per_cls)
    print(replay_dataset.offset)
    print(replay_dataset.rb_size)
    # print(len(replay_dataset.data))
    if val == True:
        eviction_idx_list = []
    num_new_label = 0
    if stream_dataset:
        for label in stream_dataset.classes_in_dataset:
            if label not in replay_dataset.offset or label not in replay_dataset.len_per_cls:
                num_new_label += 1
    if not mem_per_cls:
        num_classes = (len(replay_dataset.offset) + num_new_label)
        memory_per_cls = int(replay_dataset.rb_size / num_classes)
    else: 
        num_classes = len(replay_dataset.offset) 
        memory_per_cls = mem_per_cls
    print("MEM PER CLS : ", memory_per_cls)
    if val == False: rb_size = replay_dataset.rb_size
    old_offset, old_len_per_cls = replay_dataset.offset, replay_dataset.len_per_cls
    replay_dataset.len_per_cls = {x:memory_per_cls for x in range(num_classes)}
    replay_dataset.offset = {x:x*memory_per_cls for x in range(num_classes)}
    # skip 
    if old_offset == replay_dataset.offset and old_len_per_cls == replay_dataset.len_per_cls and stream_dataset==None: 
        if val == False: rb_size = sum(replay_dataset.len_per_cls.values())
        print(f'SKIP SAMPLING')
        return
    old_targets = [i for i in replay_dataset.targets]
    old_filenames = [i for i in replay_dataset.filenames]
    for i in range(num_classes): 
        sub_stream_data,sub_stream_label,sub_stream_filenames,sub_stream_idx = [],[],[],[]
        if stream_dataset:
            sub_stream_data,sub_stream_label,sub_stream_filenames,sub_stream_idx = stream_dataset.get_sub_data(i)
            total = (len(sub_stream_idx) + old_len_per_cls[i]) if i in old_len_per_cls else len(sub_stream_idx) 
            if memory_per_cls >= total: 
                cls_offset = total
            else: cls_offset=memory_per_cls
        else: cls_offset = memory_per_cls 
        if i < len(replay_dataset.offset)-1: 
            replay_dataset.offset[i+1] = replay_dataset.offset[i] + cls_offset
        replay_dataset.len_per_cls[i] = cls_offset      
        for j in range(cls_offset):
            st = replay_dataset.offset[i]
            # Old data triming       
            if i < (len(old_offset)) and j < old_len_per_cls[i]:
                # moving old data 
                old_st = old_offset[i]
                old_len = old_len_per_cls[i]
                new_len = replay_dataset.len_per_cls[i]
                if new_len < old_len:
                    old_idx = old_st+old_len-new_len+j
                else: 
                    old_idx = old_st + j
                replay_dataset.data[st+j] = replay_dataset.data[old_idx]
                replay_dataset.targets[st+j] = old_targets[old_idx]
                replay_dataset.filenames[st+j] = old_filenames[old_idx]
            elif stream_dataset: 
                if i not in old_offset or i not in old_len_per_cls: 
                    stream_idx = total-cls_offset+j
                else: 
                    stream_idx = total-cls_offset+j - old_len_per_cls[i]
                if (st+j)>=len(replay_dataset):
                    replay_dataset.data.append(sub_stream_data[stream_idx])
                    replay_dataset.targets.append( sub_stream_label[stream_idx])
                    replay_dataset.filenames.append(  sub_stream_filenames[stream_idx])
                else:
                    replay_dataset.data[st+j]= sub_stream_data[stream_idx]
                    replay_dataset.targets[st+j],replay_dataset.filenames[st+j] =sub_stream_label[stream_idx], sub_stream_filenames[stream_idx]
                if val == True: 
                    eviction_idx_list.extend(sub_stream_idx[-cls_offset:])
    if val == False: 
        rb_size= sum(replay_dataset.len_per_cls.values())
        print(f'REPLAY FILLED: {len(replay_dataset)}')
    print('AFTER SAMPLING')
    # print(replay_dataset.filenames)
    # print(replay_dataset.targets)
    print(replay_dataset.len_per_cls)
    print(replay_dataset.offset)
    print(replay_dataset.rb_size)
    # print(len(replay_dataset.data))
