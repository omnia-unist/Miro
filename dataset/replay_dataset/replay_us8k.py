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
from multiprocessing import shared_memory
import time
import gc
import random
EMPTY_LIST = 56 
ELEMENT_SIZE = 8 
IMG_SIZE = 100*100*3
IMG_SHAPE = (100,100,3)
IMG_DTYPE = 'uint8'

class ReplayUS8K(Dataset):
    num_file_for_label = dict()
    #num_file_for_label 서버가 종료되도 쓰일수 있게 rb_path에 따로 meta file로 저장해놔야함

    def __init__(self, rb_path, rb_size, transform, sampling, agent,dataset='urbansound8k',device='cpu',postfix=None):

        # postfix 
        if postfix == None: 
            self.postfix = os.getpid()#= self.generate_postfix()
        else: self.postfix = postfix
        self.rb_path = rb_path
        self.transform = transform
        self.test_set = dataset
        self.rb_size = rb_size
        self.filled_counter = 0
        self.filled = shared_memory.ShareableList([self.rb_size])
        self.filled[0] = 0
        self.sampling = sampling
        self.agent = agent

        self.device = device
        self.tracker = list()

        self.offset = dict()
        self.len_per_cls = dict()
        self.vec_shape = IMG_SHAPE
        self.vec_dtype =  IMG_DTYPE
        self.vec_size = IMG_SIZE
        
        self.shm_init()

    '''
    Initialize shared memory 
        One block for each vector
        One block for targets 
        One block for filenames 
        In total, rb_size + 2 blocks
    ReplayDataset should keep SELF.xxx_shm (no del or reassignment) or else the shared memory blocks will be garbage collected
    '''
    def generate_postfix(self):
        return self.postfix
    def shm_init(self): 
        self.coefficient=1
        self.data_shm=[]  
        self.data_shm_name=[]     
        
        while True: 
            try:
                self.data_shm_name = shared_memory.ShareableList([f'{self.test_set}_{str(self.device)[-1]}_replay_0000000000000000_{self.postfix}']*self.rb_size)#,name=f'replay_shm_names_{self.posifix}') 
            except (RuntimeError, FileExistsError): 
                print('ERROR')
                self.postfix=self.generate_postfix()
                continue 
            finally:
                break
        for i in range(self.rb_size):
            n=1
            while True:
                n+=1
                try:
                    item = shared_memory.SharedMemory(create=True,name=f'{self.test_set}_{str(self.device)[-1]}_replay_{i}_{self.postfix}', size=self.vec_size*self.coefficient)
                    break
                except (RuntimeError, FileExistsError): 
                    self.postfix=self.generate_postfix()
                    continue 
            b = np.ndarray(self.vec_shape, dtype=self.vec_dtype,buffer=item.buf)
            b[:] = np.full(self.vec_shape,0,self.vec_dtype)                
            self.data_shm.append(item)
            self.data_shm_name[i]=item.name   
        self.targets = shared_memory.ShareableList([-1]*self.rb_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_targets_{self.postfix}')
        self.targets_shm_name = self.targets.shm.name
        self.filenames = shared_memory.ShareableList(['0000_0000.png']*self.rb_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_filenames_{self.postfix}')
        self.filenames_shm_name = self.filenames.shm.name
    def shm_resize(self,new_size,delete=False): 
        if new_size == self.rb_size:
            return 3
        exit_code = -1
        new_mem_per_cls = int(new_size/len(self.len_per_cls))
        old_len, old_offset = self.len_per_cls,self.offset
        if new_size > len(self.data_shm):
            extra = new_size-self.rb_size
            filename_copy, targets_copy,shm_copy,  = [self.filenames[i] for i in range(len(self.filenames))], [self.targets[i] for i in range(len(self.targets))], [self.data_shm_name[i] for i in range(len(self.data_shm_name))]
            self.data_shm_name.shm.close() 
            self.data_shm_name.shm.unlink() 
            self.filenames.shm.close() 
            self.filenames.shm.unlink() 
            self.targets.shm.close() 
            self.targets.shm.unlink() 
            new_name = shared_memory.ShareableList([f'{self.test_set}_{str(self.device)[-1]}_replay_000000_{self.postfix}']*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_shm_names_{self.postfix}')
            new_files = shared_memory.ShareableList(['00000_00000.png']*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_filenames_{self.postfix}')
            new_targets = shared_memory.ShareableList([-1]*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_targets_{self.postfix}')
            for i in range(len(filename_copy)):
                new_files[i] = filename_copy[i]
                new_targets[i] = targets_copy[i]
                new_name[i] = shm_copy[i]
            self.data_shm_name,self.filenames,self.targets  = new_name,new_files,new_targets
            for i in range(len(self.data_shm),new_size): 
                create = True
                while True:
                    name = f'{self.test_set}_{str(self.device)[-1]}_replay_{i}_{self.postfix}'
                    try: 
                        item = shared_memory.SharedMemory(create=create,name=name, size=self.vec_size*self.coefficient)
                        break
                    except RuntimeError:
                        print(f'replay_shm_init postfix:{self.postfix}, sample_index: {i}, len(self.data_shm):{len(self.data_shm)}')
                        self.postfix=self.generate_postfix()
                        continue
                    except FileExistsError: #previously shirinked without delete
                        create=False
                        continue
                b = np.ndarray(self.vec_shape, dtype=self.vec_dtype,buffer=item.buf)
                b[:] = np.full(self.vec_shape,0,self.vec_dtype)                
                self.data_shm.append(item)
                self.data_shm_name[i]=item.name
            # self.data_shm_idx.extend([i for i in range(self.rb_size, new_size)])
            self.rb_size = new_size
            self.len_per_cls = {i:new_mem_per_cls for i in old_len}
            for i in range(1, len(self.offset)): self.offset[i] = self.offset[i-1] + self.len_per_cls[i-1]
            return 2
        elif new_size <= self.rb_size: 
            self.resize_sampling(new_mem_per_cls)
            for i in range(new_size,len(self.filenames)):
                self.targets[i] = -1
            self.rb_size = new_size
            if new_size <= self.filled[0]:
                self.filled[0] = new_size 
            exit_code = 0 
        elif new_size <= len(self.data_shm): 
            self.rb_size = new_size
            self.filled[0] = new_size
            exit_code = 1
        if delete == True: 
            # reassign lists
            filename_copy, targets_copy,shm_copy,  = [self.filenames[i] for i in range(new_size)], [self.targets[i] for i in range(new_size)], [self.data_shm_name[i] for i in range(new_size)]
            self.data_shm_name.shm.close() 
            self.data_shm_name.shm.unlink() 
            self.filenames.shm.close() 
            self.filenames.shm.unlink() 
            self.targets.shm.close() 
            self.targets.shm.unlink() 
            new_name = shared_memory.ShareableList([f'{self.test_set}_{str(self.device)[-1]}_replay_000000_{self.postfix}']*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_shm_names_{self.postfix}')
            new_files = shared_memory.ShareableList(['00000_00000.png']*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_filenames_{self.postfix}')
            new_targets = shared_memory.ShareableList([-1]*new_size,name=f'{self.test_set}_{str(self.device)[-1]}_replay_targets_{self.postfix}')
            for i in range(len(filename_copy)):
                new_name[i],new_files[i],new_targets[i] = shm_copy[i],filename_copy[i],targets_copy[i]
            self.data_shm_name,self.filenames,self.targets  = new_name,new_files,new_targets
            #delete data points
            for i in range(new_size,len(self.data_shm)):
                self.clean(i) 
            self.data_shm = self.data_shm[:new_size]
            if exit_code == 1:
                self.resize_sampling(new_mem_per_cls)
        gc.collect()
        return exit_code
    def resize_sampling(self, mem_per_cls=None):
        
        num_classes = len(self.offset) 
        budget = mem_per_cls * num_classes
        new_len_per_cls =  {x:self.len_per_cls[x] for x in range(num_classes)}
        smaller_classes = {label:n_samples for i,(label,n_samples) in enumerate(self.len_per_cls.items()) if n_samples<mem_per_cls}
        smaller_labels = []
        while len(smaller_classes)>0:
            new_len_per_cls.update(smaller_classes)
            smaller_labels.extend(list(smaller_classes.keys()))
            budget -= sum(smaller_classes.values())
            num_classes  = len(self.offset)- len(smaller_labels)
            if num_classes ==0: break
            mem_per_cls = budget//num_classes # Originaly was just deviding causing float value to appear, changed to only return in, could this have problems?
            smaller_classes = {label:n_samples for  i,(label,n_samples) in enumerate(self.len_per_cls.items()) if n_samples<mem_per_cls and label not in smaller_labels} 
             

        memory_per_cls = mem_per_cls
        print("MEM PER CLS : ", memory_per_cls)
        # temporarily lift length limit 
        self.filled[0] =self.rb_size
        
        old_offset, old_len_per_cls = self.offset, self.len_per_cls
        self.len_per_cls = new_len_per_cls
        for i in range(1, len(self.offset)): self.offset[i] = self.offset[i-1] + self.len_per_cls[i-1]
        old_targets = [i for i in self.targets]
        old_filenames = [i for i in self.filenames]
        
        for i in range(num_classes): 
            cls_offset = memory_per_cls 
            if i < len(self.offset)-1: 
                self.offset[i+1] = self.offset[i] + cls_offset
            self.len_per_cls[i] = cls_offset      
            for j in range(cls_offset):
                st = self.offset[i]
                # Old data triming       
                if i < (len(old_offset)) and j < old_len_per_cls[i]:
                    # moving old data 
                    old_st = old_offset[i]
                    old_len = old_len_per_cls[i]
                    new_len = self.len_per_cls[i]
                    if new_len < old_len:
                        old_idx = old_st+old_len-new_len+j
                    else: 
                        old_idx = old_st + j
                    
                    old_data_shm = self.data_shm[old_idx]
                    old_data = np.ndarray(self.vec_shape,self.vec_dtype,buffer=old_data_shm.buf)
                    new_data_shm = self.data_shm[st+j]
                    new_data = np.ndarray(self.vec_shape,self.vec_dtype,buffer=new_data_shm.buf)
                    new_data[:] = old_data[:]
                    self.targets[st+j] = old_targets[old_idx]
                    self.filenames[st+j] = old_filenames[old_idx]
        self.filled[0] = sum(self.len_per_cls.values())


    '''
    returns the open shared memory blocks. Send this to SwapManager and Saver to allow their access to replay dataset 
    '''
    def get_shm_names(self):
        return self.data_shm_name.shm.name, self.targets_shm_name, self.filenames_shm_name
    
    '''
    Close and Unlink all the shared memory blocks 
    '''
    def clean(self,idx):
        self.data_shm[idx].close()
        self.data_shm[idx].unlink()
        self.data_shm[idx] =None
    def cleanup(self):
        for shm in self.data_shm: 
            shm.close()
            # shm.unlink()
        self.data_shm_name.shm.close()
        self.targets.shm.close() 
        # self.targets.shm.unlink()
        self.filenames.shm.close()
        # self.filenames.shm.unlink()
        self.filled.shm.close()
    
    def __len__(self):
        return self.filled[0]
        
    def is_filled(self):
        if self.filled[0] <= 0:
            return False
        else:
            return True
    def __setitem__(self,idx,value):
        
        # Parsing 
        if type(value) == tuple:
            try: vec = value[0]
            except Exception: raise "replay_dataset.__setitem__: invalid value"
            try: label = value[1]
            except Exception: label=None
            try: filename = value[2]
            except Exception: filename=None 

        else: vec,filename,label = value,None,None

        old_idx = idx
        # idx = self.data_shm_idx[idx]
        if type(idx) == int: 
            if old_idx >= self.filled[0]:
                raise "Replay idx out of bound"
            vec = np.array(vec)
            data_shm = self.data_shm[idx]
            if vec.size > data_shm.size: 
                print(f'expand shared memory {data_shm} -> {vec.size}')
                self.clean(idx)
                self.data_shm[idx] = shared_memory.SharedMemory(create=True, name=f'{self.test_set}_{str(self.device)[-1]}_replay_{idx}_{self.postfix}',size=vec.size)
                self.data_shm_name[idx] = self.data_shm[idx].name
                data_shm = self.data_shm[idx]
            data = np.ndarray(self.vec_shape,self.vec_dtype,buffer=data_shm.buf)
            data[:] = vec
            if filename is not None: 
                self.filenames[old_idx] = filename
            if label is not None: 
                self.targets[old_idx] = label
                
    def __getitem__(self, idx):
        return self.getitem_offline(idx)
    '''
    Manually keep track of whether idx is out of bound, for LEN(REPLAY) will constantly equal to RB_SIZE
    '''
    def getitem_offline(self, idx):
        old_idx = idx
        if type(idx) == int:  
            if old_idx >= len(self):
                print(f'idx: {old_idx}',end=' ')
                raise "Replay idx out of bound"
            data_shm = self.data_shm[idx]
            data = np.ndarray(self.vec_shape,self.vec_dtype,buffer=data_shm.buf)
            
            if self.transform is not None:
                data = self.transform(data)
            else:
                data = data
            target = self.targets[old_idx]
            filename = self.filenames[old_idx]
            return old_idx, data, target, filename
        # Get sublist 
        if isinstance(idx,slice): 
            dlist = []
            idxs = []
            for i in range(*idx.indices(self.filled[0])): 
                data = self[i][0]
                dlist.append(torch.from_numpy(data))
                idxs.append(i)
            return idxs, dlist,[self.targets[ii] for ii in range(*idx.indices(len(self)))], [self.filenames[ii] for ii in range(*idx.indices(len(self)))]
    
    def get_meta(self): 
        meta = dict()
        meta['agent'] = self.agent 
        meta['rb_size'] = self.rb_size
        meta['rp_len_name'] = self.filled.shm.name
        meta['rb_path'] = self.rb_path
        meta['transform'] = self.transform
        
        meta['vec_size'] = self.vec_size
        meta['vec_dtype'] = self.vec_dtype
        meta['vec_shape'] = self.vec_shape
        
        meta['img_size'] = self.vec_size
        meta['img_dtype'] = self.vec_dtype
        meta['img_shape'] = self.vec_shape

        meta['test_set'] = 'urbansound8k' 
        meta['data_shm_name'],meta['targets_shm_name'],meta['filenames_shm_name'] = self.get_shm_names()
        return meta      
