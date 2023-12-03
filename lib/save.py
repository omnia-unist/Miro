import os
from array import array
import asyncio
import pickle

import queue
import threading
# import multiprocessing as python_multiprocessing
import torch.multiprocessing as python_multiprocessing
from multiprocessing import shared_memory
import directio, sys, mmap, io
import math
import numpy as np
import random
import torch
import gc

class DataSaver(object):


    def __init__(self, rb_path, store_budget=None, seed=None,dataset='cifar'):
        self.testset = dataset
        
        if len(rb_path)-1 == '/':
            rb_path = rb_path[:len(rb_path)-1]
        self.rb_path = rb_path
        self.store_budget = store_budget
        num_labels = {'cifar100':100, 'tiny_imagenet':200, 'urbansound8k':10, 'imagenet1000':1000,'audioset':100, 'dailynsports':19}
        self.num_file_for_label_for_swap = [[-1]*num_labels[dataset]]
        self.num_file_for_label= [shared_memory.ShareableList([-1]*num_labels[dataset])]
        self.seed = seed
        if dataset in ["twentynews", "dailynsports", "shakespeare","audioset"]: self.png_save = False
        else: self.png_save = True
        
    def _translate_idx(self,label):
        idx1,idx2 = int(label/len(self.num_file_for_label[0])), label%len(self.num_file_for_label[0])
        return idx1, idx2
    def label_in_num_file_for_label(self,label): 
        idx1,idx2 = self._translate_idx(label)
        return self.num_file_for_label[idx1][idx2]>=0
    def get_num_file_for_label(self, label):
        assert(self.label_in_num_file_for_label(label))
        idx1,idx2 = self._translate_idx(label)
        return self.num_file_for_label[idx1][idx2]
    def get_num_file_for_label_for_swap(self, label):
        idx1,idx2 = self._translate_idx(label)
        return self.num_file_for_label_for_swap[idx1][idx2]
    def inc_num_file_for_label(self, label,value):
        idx1,idx2 = self._translate_idx(label)
        self.num_file_for_label[idx1][idx2] += value
    def dec_num_file_for_label(self, label,value):
        idx1,idx2 = self._translate_idx(label)
        self.num_file_for_label[idx1][idx2] -= value
    def num_class_stored(self): 
        res = 0
        for part in self.num_file_per_label: 
            res += len(part) - part.count(-1)
        return res
    def before_train(self):
        self.save_done_event = python_multiprocessing.Event()
        self.save_queue = python_multiprocessing.Queue()
        self.save_worker = python_multiprocessing.Process(
                    target = self.save_loop,
                    args=(self.save_queue, self.save_done_event,self.seed)
        )
        self.save_worker.daemon = True
        self.save_worker.start()
        
    def after_train(self):
        
        self.save_done_event.set()
        self.save_queue.put((None,None,None,None))
        self.save_worker.join()
        self.save_queue.cancel_join_thread()
        self.save_queue.close()

        gc.collect()
        # notify saver arrival of new data 
        for i in range(len(self.num_file_for_label_for_swap)):
            for j in range(len( self.num_file_for_label_for_swap[i])):
                self.num_file_for_label_for_swap[i][j] = self.num_file_for_label[i][j]
    async def makedir(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception:
            return False

    #direct I/O
    async def img_save(self, img, path, label, logit=None, logit_path=None):
        
        imgByteArr = io.BytesIO()
        if self.dataset in ['audioset']:
            np.save(imgByteArr, img)
        else:
            img.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        f = os.open(path, os.O_RDWR | os.O_CREAT | os.O_DIRECT)

        block_size = 512 * math.ceil(sys.getsizeof(imgByteArr) / 512)
        m = mmap.mmap(-1, block_size)
        m.write(imgByteArr)

        ret = directio.write(f, m[:block_size])

        os.close(f)

        if logit is not None and logit_path is not None:
            logit_byte = pickle.dumps(logit)
            f = os.open(logit_path, os.O_RDWR | os.O_CREAT | os.O_DIRECT)
                
            block_size = 512 * math.ceil(sys.getsizeof(logit_byte) / 512)
            m = mmap.mmap(-1, block_size)
            m.write(logit_byte)

            ret = directio.write(f, m[:block_size])
            os.close(f)
    
        
        return label


    def wait_img_save(self, img, path, label, logit=None, logit_path=None):
        
        imgByteArr = io.BytesIO()
        if self.testset in ['audioset']:
            np.save(imgByteArr, img)
        else:
            img.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        f = os.open(path, os.O_RDWR | os.O_CREAT | os.O_DIRECT)

        block_size = 512 * math.ceil(sys.getsizeof(imgByteArr) / 512)
        m = mmap.mmap(-1, block_size)
        m.write(imgByteArr)

        ret = directio.write(f, m[:block_size])

        os.close(f)

        if logit is not None and logit_path is not None:
            logit_byte = pickle.dumps(logit)
            f = os.open(logit_path, os.O_RDWR | os.O_CREAT | os.O_DIRECT)
                
            block_size = 512 * math.ceil(sys.getsizeof(logit_byte) / 512)
            m = mmap.mmap(-1, block_size)
            m.write(logit_byte)

            ret = directio.write(f, m[:block_size])
            os.close(f)
    
        return label

    

    async def main_budget(self, stream_data, stream_targets, stream_outputs=None,filename=None):

        for i, (data, label) in enumerate(zip(stream_data, stream_targets)):        
            label_path = self.rb_path + '/' + str(label)
            if not self.label_in_num_file_for_label(label):
                self.inc_num_file_for_label(label,1)
                await self.makedir(label_path)
            
            storage_per_cls = self.store_budget // self.num_class_stored()

            for part in self.num_file_for_label: 
                for i in range(len(part)): 
                    if part[i] <0: 
                        continue 
                    del_label_path = self.rb_path + '/' + str(la)
                    if storage_per_cls <= part[i]:
                        del_st = storage_per_cls 
                        del_en = part[i] 
                        for del_file in range(del_en, del_st-1,-1):
                            del_filepath = del_label_path + '/' + str(del_file) + '.png' 
                            os.remove(del_filepath)
                            part[i] -= 1
                    
            if storage_per_cls == self.get_num_file_for_label(label):
                del_filepath = label_path + '/' + str(self.get_num_file_for_label(label)) + '.png'
                os.remove(del_filepath)
                self.dec_num_file_for_label(label,1)

            assert storage_per_cls > self.get_num_file_for_label(label)

            curr_num = str(self.num_file_for_label(label) + 1)
            
            if self.png_save: file_name = curr_num + ".png"
            else: file_name = curr_num + ".npy"
            file_path = label_path + '/' + file_name  

            if not os.path.exists(file_path): # ONLY ONCE
                completed_label = label
            elif not self.png_save: 
                np.save(file_path, data)
                completed_label = label
            elif stream_outputs is not None:
                logit_file_name = curr_num + ".pkl"
                logit_file_path = label_path + '/' + logit_file_name
                logit = stream_outputs[i]
                completed_label = await self.img_save(data, file_path, label, logit, logit_file_path)
            else:
                completed_label = await self.img_save(data, file_path, label)

            if completed_label is not False:
                self.inc_num_file_for_label(completed_label,1)
            del data, label

    async def main(self, stream_data, stream_targets, filename=None, stream_outputs=None):
        if (filename is None):
            for i, (data, label) in enumerate(zip(stream_data, stream_targets)):
                label_path = self.rb_path + '/' + str(label)
                if not self.label_in_num_file_for_label(label):
                    self.inc_num_file_for_label(label,1)
                    await self.makedir(label_path)
            
                
                curr_num = str(self.get_num_file_for_label(label) + 1)

                if self.png_save: file_name = curr_num + ".png"
                else: file_name = curr_num + ".npy"
                file_path = label_path + '/' + file_name


                if not os.path.exists(file_path): # ONLY ONCE
                    completed_label = label
                elif not self.png_save: 
                    np.save(file_path, data)
                    completed_label = label
                elif stream_outputs is not None:
                    logit_file_name = curr_num + ".pkl"
                    logit_file_path = label_path + '/' + logit_file_name
                    logit = stream_outputs[i]
                    completed_label = await self.img_save(data, file_path, label, logit, logit_file_path)
                else:
                    completed_label = await self.img_save(data, file_path, label)

                if completed_label is not False:
                    self.inc_num_file_for_label(completed_label,1)
                del data, label
        else:
            for i, (data, label, name) in enumerate(zip(stream_data, stream_targets, filename)):
                label_path = self.rb_path + '/' + str(label)
                if not self.label_in_num_file_for_label(label):
                    self.inc_num_file_for_label(label,1)
                    await self.makedir(label_path)
                curr_num = str(name)

                if self.png_save: file_name = curr_num + ".png"
                else: file_name = curr_num + ".npy"
                file_path = label_path + '/' + file_name  
                if not os.path.exists(file_path): # ONLY ONCE
                    completed_label = label
                elif not self.png_save: 
                    np.save(file_path, data)
                    completed_label = label
                elif stream_outputs is not None:
                    logit_file_name = curr_num + ".pkl"
                    logit_file_path = label_path + '/' + logit_file_name
                    logit = stream_outputs[i]
                    completed_label = await self.img_save(data, file_path, label, logit, logit_file_path)
                else:
                    completed_label = await self.img_save(data, file_path, label)

                if completed_label is not False:
                    self.inc_num_file_for_label(completed_label,1)
                del data, label

    def wait_main(self, stream_data, stream_targets, filename=None, stream_outputs=None):
        completed_label = False
       
        if (filename is None):
            for i, (data, label) in enumerate(zip(stream_data, stream_targets)):
                label_path = self.rb_path + '/' + str(label)
                if not self.label_in_num_file_for_label(label):
                    self.inc_num_file_for_label(label,1)
                
                if not os.path.exists(label_path): 
                    os.makedirs(label_path, exist_ok=True)
            
                curr_num = str(self.get_num_file_for_label(label) + 1)

                if self.png_save: file_name = curr_num + ".png"
                else: file_name = curr_num + ".npy"
                file_path = label_path + '/' + file_name

                if os.path.exists(file_path): # ONLY ONCE
                    print('file exists', end=' ')
                    completed_label = label
                elif not self.png_save: 
                    np.save(file_path, data)
                    completed_label = label
                elif stream_outputs is not None:
                    logit_file_name = curr_num + ".pkl"
                    logit_file_path = label_path + '/' + logit_file_name
                    logit = stream_outputs[i]
                    completed_label = self.wait_img_save(data, file_path, label, logit, logit_file_path)
                else:
                    completed_label = self.wait_img_save(data, file_path, label)

                if completed_label is not False:
                    self.inc_num_file_for_label(completed_label,1)
                del data, label
        else:
            
            for i, (data, label, name) in enumerate(zip(stream_data, stream_targets, filename)):
                label_path = self.rb_path + '/' + str(label)
                if not self.label_in_num_file_for_label(label):
                    self.inc_num_file_for_label(label,1)


                if not os.path.exists(label_path): 
                    os.makedirs(label_path, exist_ok=True)
                
                curr_num = str(name)

                if self.png_save: file_name = curr_num + ".png"
                else: file_name = curr_num + ".npy"
                file_path = label_path + '/' + file_name  

                if os.path.exists(file_path): # ONLY ONCE
                    completed_label = label
                elif not self.png_save: 
                    np.save(file_path, data)
                    completed_label = label
                elif stream_outputs is not None:
                    logit_file_name = curr_num + ".pkl"
                    logit_file_path = label_path + '/' + logit_file_name
                    logit = stream_outputs[i]
                    completed_label = self.wait_img_save(data, file_path, label, logit, logit_file_path)
                else:
                    completed_label = self.wait_img_save(data, file_path, label)

                if completed_label is not False:
                    self.inc_num_file_for_label(completed_label,1)
                del data, label


    def save_loop(self, save_queue, save_done_event, seed=None):

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        while True:
            stream_data, stream_targets, filename, stream_outputs = self.save_queue.get()
            if stream_data == None:
                assert save_done_event.is_set()
                break
            if save_done_event.is_set():
                continue

            if (self.store_budget is not None) and (filename is None):
                asyncio.run(self.main_budget(stream_data, stream_targets, stream_outputs))
            else:
                asyncio.run(self.main(stream_data, stream_targets, filename, stream_outputs))
            
            del stream_data, stream_targets, filename, stream_outputs
    def save(self, stream_data, stream_targets, filename=None, stream_outputs=None):
        # asyncio.run(self.main(stream_data, stream_targets, stream_outputs))
        
        # self.save_queue.put((stream_data, stream_targets, filename, stream_outputs))
        #print("ALL DATA IS SAVED!!")
        
               
        # if self.testset in 'imagenet':
            # print("SAVING ALL STREAM SAMPLES...")
        self.wait_main(stream_data, stream_targets, filename, stream_outputs)
        
        
        # else:
        # print("SAVING ALL STREAM SAMPLES...WORKER")
        # self.save_queue.put((stream_data, stream_targets, filename, stream_outputs))
        