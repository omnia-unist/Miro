
from turtle import down
from torch.utils.data import Dataset
import numpy as np

from set_dataset import Continual
from types import SimpleNamespace
import os
import torch
import random

JETSON=False

def load_data(fpath,task_order=None):
    if task_order == '/fixed/':
        fpath = fpath + task_order+'/index.txt'
        task_st, class_st = 0,0
        data = []
        labels = []
        for x in range (10):
            data.append([])
            labels.append([])
        lines = open(fpath)
        # path_prefix = '/data/project/rw/'
        for line in lines:
            arr = line.strip().split()
            # path =path_prefix+path
            label = int(arr[1])
            # when a full set is not used
            if(label>=len(labels)):
                break
            data[label].append(arr[0])
            labels[label].append(label)
        return data, labels, None
    else: 
        fpath = fpath + task_order 
        flist = os.listdir(fpath)
        task_size = len(flist)-1
        # task_size = 3
        data, labels,metadata= [],[],[]
        for i in range(task_size): 
            data.append([])
            labels.append([])
            metadata.append([])
        flist = sorted(os.listdir(fpath))
        for idx_file in flist:
            if idx_file == 'test.txt': continue
            task_data,task_labels = {},{}
            task_id = int(idx_file.split('.txt')[0][4:])-1
            if task_id >= task_size: break
            with open(f'{fpath}/{idx_file}') as lines: 
                for line in lines: 
                    arr = line.strip().split()
                    # path =path_prefix+path
                    label = int(arr[1])
                    if label not in task_labels: 
                        task_data[label],task_labels[label] = [],[]
                    task_data[label].append(arr[0])
                    task_labels[label].append(label)
            task_meta = {label:len(task_labels[label]) for label in task_labels}
            data[task_id] = (task_data)
            labels[task_id] = task_labels
            metadata[task_id] =task_meta
        print(metadata)     
        return data,labels,metadata
def experiment(final_params):
    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        
        if hasattr(final_params, 'rb_path'):
            org_rb_path = final_params.rb_path
            print(final_params.rb_path)

        # elif hasattr(final_params, 'rb_path'):
        #     final_params.rb_path = org_rb_path + '/' + f'{final_params.filename}'
        #     os.makedirs(final_params.rb_path, exist_ok=True)

        #     print(final_params.rb_path)
        print(final_params.filename)

        
        
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)


        
        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)
                
                final_params.seed = seed

        num_task_cls_per_task = final_params.num_task_cls_per_task
        # pass samples per class, samples per task to Continual object 
        final_params.samples_per_cls = [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000]
        final_params.samples_per_task = []

        class_order = None
        task_order = None 
        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        elif final_params.data_order == 'fixed':
            # class_order = [0,1,2,3,4,5,6,7,8,9]
            # class_order = [4,2,7,3,9,1,5,6,8,0,]
            # {0: 64.0, 1: 72.0, 2: 66.0, 3: 60.0, 4: 64.0, 5: 62.0, 6: 98.0, 7: 72.0, 8: 94.0, 9: 94.0}
            # class_order = [9,1,2,4,0,3,5,7,6,8]
            class_order = [6,4,0,1,2,3,5,7,8,9]
            # class_order = [5,6,4,0,1,2,3,7,8,9]
            task_order = '/fixed/'
        elif final_params.data_order == 'blurry1':
            task_order = '/blurry1_index/'
        elif final_params.data_order == 'blurry2':
            task_order = '/blurry2_index/'
        elif final_params.data_order == 'blurry3':
            task_order = '/blurry3_index/'
        elif final_params.data_order == 'non-blurry1':
            task_order = '/non-blurry1_index/'
        elif final_params.data_order == 'non-blurry2':
            task_order = '/non-blurry2_index/'
        else: task_order = f'/{final_params.data_order}/'
        fpath = final_params.test_set_path 
        # Task Partition: 
        if class_order is not None:
            data, labels,metadata = load_data(fpath,task_order) # 100tasks, 10cls per task print(class_order)
            assert((sum([a*b for (a,b) in (num_task_cls_per_task)]))<=len(class_order))
            task_st, class_st = 0,0
                
            for (num_task, cls_per_task) in num_task_cls_per_task:
                for task_id in range (task_st,task_st+num_task):
                    samples_per_task = 0
                    print(f'    Task_{task_id}: ',end = '[')
                    for x in range (class_st, class_st + cls_per_task):
                        idx_of_cls = class_order[x]
                        samples_per_task += final_params.samples_per_cls[idx_of_cls]
                        print(f'{idx_of_cls},',end='')
                    print(']')
                    class_st+=cls_per_task  
                    final_params.samples_per_task.append(samples_per_task)
                task_st += num_task
            print(final_params.samples_per_task)
            assert(sum([task*cls_num for (task,cls_num) in num_task_cls_per_task])<=10)
            continual = Continual(**vars(final_params))
            print('AGENT READY', flush=True)
            from PIL import Image
            task_st, class_st = 0,0
            # data = data[0]
            
            if JETSON: 
                import power_check as pc
                pl = final_params.power_log
                
            for (num_task, cls_per_task) in num_task_cls_per_task:
                for task_id in range (task_st,task_st+num_task):
                    if JETSON: 
                        pc.printFullReport(pl.device)
                        pl.recordEvent(name='New Task Start')
                        
                    for label in range (class_st, class_st + cls_per_task):
                        for data_path in data[class_order[label]]:
                            data_path = os.path.join(os.path.abspath(final_params.test_set_path), data_path)
                            with open(data_path,'rb') as f:
                                # if (cnt%100==0):print(f)
                                file = Image.open(f)
                                img = file.convert('RGB')
                            continual.send_stream_data(img, class_order[label], task_id)
                            del img,file
                    
                    class_st+=cls_per_task  
                    continual.samples_per_task = cls_per_task*final_params.samples_per_cls[task_id]
                    continual.samples_per_cls_per_task = final_params.samples_per_cls[task_id]
                    continual.train_disjoint(task_id)
                task_st += num_task
        else: 
            data, labels,metadata = load_data(fpath,task_order) # 100tasks, 10cls per task
            print(task_order,metadata)
            final_params.samples_per_task = [sum(d.values()) for d in metadata]
            continual = Continual(**vars(final_params))
            continual.samples_per_cls_per_task = metadata
            print(f'samples_per_task: {continual.samples_per_task }')
            from PIL import Image
            
            if JETSON: 
                import power_check as pc
                pl = final_params.power_log
                
            for task_id in range(len(data)): 
                if JETSON: 
                    pc.printFullReport(pl.device)
                    pl.recordEvent(name='New Task Start')
                        
                for label in data[task_id]: 
                    # print(f'Task{task_id+1}, labels{label}, {len(data[task_id][label])} samples')
                    for data_path in data[task_id][label]: 
                        with open(data_path,'rb') as f: 
                            file = Image.open(f)
                            img = file.convert('RGB')
                        continual.send_stream_data(img, label, task_id)
                        del img, file
                print(len(continual.agent.stream_dataset))
                continual.train_disjoint(task_id)
        
        if JETSON: 
            pc.printFullReport(pl.device)
            pl.recordEvent(name='All Task End')
            
        # Close Shared Memory 
        continual.agent.replay_dataset.cleanup()
        f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
        f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
        f1.write(f'Total rounds of swapping: {len(continual.agent.num_swap)}\n')
        f1.close()
        print(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
