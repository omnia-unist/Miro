from torch.utils.data import Dataset
import numpy as np
from set_dataset import Continual
from types import SimpleNamespace
import os
import torch
import random
import sys

class iDailynSports(Dataset):
    """
    Defines DailynSports as for the other pytorch datasets.
    """
    def __init__(self, dataset_path: str='data') -> None:
        self.data = []
        self.targets = []
        
        # load the data from the directory
        for a_num in range(1,20): # 19 categories
            for p_num in range(1,9): # 8 users 
                for s_num in range(1,49):#range_txt: # 1-48 train, 49-60 test
                    np_data = np.loadtxt(str(dataset_path)+"/a"+str(a_num).zfill(2)+"/p"+str(p_num).zfill(1)+"/s"+str(s_num).zfill(2)+".txt",dtype=np.float32,delimiter=",")
                    self.data.append(np_data) # dimension = (125,45)
                    self.targets.append(a_num)
                        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        # Carriers for train data & label
        self.TrainData = []
        self.TrainLabels = []
        
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTrainData(self,label):
        datas,labels=[],[]

        data=self.data[self.targets==label]
        datas.append(data)
        labels.append(np.full((data.shape[0]),label))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)

    def __len__(self):
        return len(self.TrainData)

    def __getitem__(self, index):
        vec, target = self.TrainData[index], self.TrainLabels[index]
        return vec,target

def load_data(fpath,task_order=None):
    if task_order == 'fixed':
        fpath = fpath + '/index.txt'
        data = []
        labels = []

        for _ in range (10):
            data.append([])
            labels.append([])
        lines = open(fpath)

        for line in lines:
            arr = line.strip().split()
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
        data, labels,metadata= [],[],[]
        for _ in range(task_size): 
            data.append([])
            labels.append([])
            metadata.append([])
        for idx_file in os.scandir(fpath):
            if idx_file.name == 'test.txt': continue
            task_data,task_labels = {},{}
            task_id = int(idx_file.name.split('.txt')[0][4:])-1
            with open(idx_file) as lines: 
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
        return data,labels,metadata

def experiment(final_params):
    # Run the Experiment for number of "run" in final_params
    runs = final_params.run
    for num_run in range(runs):
        print(f"#RUN{num_run}")

        # Set filename
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        final_params.filename = org_filename + f'run{num_run}'

        # Print out param infos, set basic parameters as seeds
        if hasattr(final_params, 'rb_path'):
            print(f'Storage Path: {final_params.rb_path}')
        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
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


        # pass samples per class, samples per task to Continual object 
        final_params.samples_per_cls = 480 
        final_params.samples_per_task = []

        class_order = None

        # Set class order for class incrmental learning
        if final_params.data_order == 'seed':
            class_order = np.arange(19)
            np.random.shuffle(class_order)
        elif final_params.data_order == 'fixed':
            # class_order = [7, 4, 18, 6, 14, 10, 2, 11, 3, 8, 15, 17, 5, 19, 13, 1, 16, 12, 9]
            class_order = [i for i in range(1,20)]
            # class_order = [20 - i for i in range(1,20)]
            
        else: 
            print("Currently Not Implemented / Deprecated Error")
            sys.exit()
        
        # Check if the devision of input classes into class order is posible
        num_task_cls_per_task = final_params.num_task_cls_per_task
        # assert(sum([task*cls_num for (task,cls_num) in num_task_cls_per_task])<=len(class_order))
            
        # Task Partition: 
        if class_order is not None:
            # Create a dataset (= load data)
            print("[Preparing Dataset] DSADS...", end='  ')
            dataset = iDailynSports(final_params.test_set_path)
            
            # Partition the tasks
            print('Task Partition:')
            task_st, class_st = 0,0
            for (num_task, cls_per_task) in num_task_cls_per_task:
                for task_id in range (task_st,task_st+num_task):
                    if task_id == 9:
                        cls_per_task=1
                    else:
                        cls_per_task=2
                    print(f'    Task_{task_id}: ',end = '[')
                    for x in range (class_st, class_st + cls_per_task):
                        print(f'{class_order[x]},',end='')
                    print(']')
                    class_st+=cls_per_task  
                    final_params.samples_per_task.append(cls_per_task*final_params.samples_per_cls)
                task_st += num_task
            print(final_params.samples_per_task)

            # Create the continual object
            continual = Continual(**vars(final_params))
            if final_params.jetson: 
                import power_check as pc
                pl = final_params.power_log
                
            [task_st, class_st] = [0,0]
            for (num_task, cls_per_task) in num_task_cls_per_task:
                for task_id in range (task_st,task_st+num_task):
                    if final_params.jetson: 
                        pc.printFullReport()
                        pl.recordEvent(name='New Task Start')
                    
                    if task_id == 9:
                        cls_per_task=1
                    else:
                        cls_per_task=2
                    for x in range (class_st, class_st + cls_per_task):
                        dataset.getTrainData(class_order[x])
                        for i in range(len(dataset)):
                            vec,label = dataset[i]
                            continual.send_stream_data(vec,label,task_id)
                    class_st+=cls_per_task  
                    continual.samples_per_task = cls_per_task*final_params.samples_per_cls
                    continual.train_disjoint(task_id)
                task_st += num_task

        if final_params.jetson: 
            pc.printFullReport()
            pl.recordEvent(name='All Task End')
        # Close Shared Memory 
        continual.agent.replay_dataset.cleanup()
        f1 = open(final_params.result_save_path + final_params.filename + '_accuracy.txt', 'a')
        f1.write(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
        f1.write(f'Total rounds of swapping: {len(continual.agent.num_swap)}\n')
        f1.close()
        print(f'Total number of swapping: {sum(continual.agent.num_swap)}\n')
