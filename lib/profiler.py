import psutil as ps
from collections import namedtuple
class Profiler(object):
    def __init__(self,unit='mb'):
        self.processes = {'Main':ps.Process()}
        self.unit = {'kb':1024,'mb':1024**2,'gb':1024**3}
        self.pmem = {'kb':namedtuple('pmem_kb', 'used virtual shared unique pss text lib data'),'mb':namedtuple('pmem_mb', 'used virtual shared unique pss text lib data'),'gb':namedtuple('pmem_gb', 'used virtual shared unique pss text lib data')}
        self.mem_log = {x:[] for x in ['Main', 'Swapper', 'Saver', 'Dataloader']}
    '''
    task_id: specific task number you wish to look at. If task_id == None, accumulated results will be returned 
    worker_name: 'Main', 'Swapper' ,'Saver' ,'Dataloader' ,'Combined' 
    '''
    def add_worker(self, worker_name, pids,task_id):
        if type(pids) is not list: pids = [pids]
        assert (worker_name in ['Main', 'Swapper', 'Saver', 'Dataloader', 'Combined' ])
        if worker_name in self.processes: 
            if len(self.processes[worker_name]) <= task_id: 
                self.processes[worker_name].extend([None] *(task_id-len(self.processes[worker_name])) )
                self.processes[worker_name].append(pids)
            else: 
                self.processes[worker_name][task_id].extend(pids)
        else:
            self.processes[worker_name] = [None] * task_id
            self.processes[worker_name].append(pids)
            
            
    def record_memory(self, worker_name='Main',unit='mb'): 
        assert (worker_name in ['Main', 'Swapper', 'Saver', 'Dataloader', 'Combined' ])
        p = self.processes[worker_name]
        munit = self.unit[unit]
        if type(p) is not list:
            psmem = p.memory_full_info()
            pmem_format =  self.pmem[unit]
            pmem = pmem_format(psmem.rss/munit,
                                psmem.vms/munit,
                                psmem.shared/munit,
                                psmem.uss/munit,
                                psmem.pss/munit,
                                psmem.text/munit,
                                psmem.lib/munit,
                                psmem.data/munit)
            self.mem_log[worker_name].append(pmem)
            return True
        else:
            local_list=[]
            for pid in p[-1]:
                process = ps.Process(pid)
                psmem = process.memory_full_info()
                pmem_format =  self.pmem[unit]
                pmem = pmem_format(psmem.rss/munit,
                                    psmem.vms/munit,
                                    psmem.shared/munit,
                                    psmem.uss/munit,
                                    psmem.pss/munit,
                                    psmem.text/munit,
                                    psmem.lib/munit,
                                    psmem.data/munit)
                local_list.append(pmem)
            self.mem_log[worker_name].append(local_list)
    def get_memory(self,task_id=None, worker_name='Combined'):
        assert (worker_name in ['Main', 'Swapper', 'Saver', 'Dataloader', 'Combined' ])

        if worker_name == 'Combined':
            if task_id is None: return self.mem_log
            else: 
                return [log[task_id] for log in self.memlog]
        else:
            if task_id is None:                     
                return self.mem_log[worker_name]
            else: return self.mem_log[worker_name][task_id]
    
# pmem(rss=99024896, vms=343052288, shared=37838848, text=2424832, lib=0, data=61706240, dirty=0)
        #used, virtual, shared, text,lib,data,unique,pss
# import pyRAPL
# import math
# pyRAPL.setup() 
# measure = pyRAPL.Measurement('bar')
# measure.begin()
# def foo():
#     for i in range(10000):
#         a = i**2

# foo()
# measure.end()

#Result(label='bar', timestamp=1668773946.673289, duration=20470.478, pkg=[1337033.0, 1274350.0], dram=[123256.0, 196590.0]) # 100000
# Result(label='bar', timestamp=1668774014.7942364, duration=2100.149, pkg=[136109.0, 217773.0], dram=[18468.0, 28091.0])  #10000