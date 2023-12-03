#!/usr/bin/env python3
"""
Convient power measurement script for the Jetson TX2/Tegra X2. 
relevant docs: http://developer2.download.nvidia.com/embedded/L4T/r27_Release_v1.0/Docs/Tegra_Linux_Driver_Package_Release_Notes_R27.1.pdf
@author: Lukas Cavigelli (cavigelli@iis.ee.ethz.ch)
"""

import sys
sys.path.append("/home/omnia/.local/lib/python3.8/site-packages")
sys.path.append("/home/omnia/.local/lib/python3.8/site-packages/torchvision-0.13.0-py3.8-linux-aarch64.egg")
device_nodes = {
    'jetson_tx2':[('module/main' , '0041', '0'),
         ('module/gpu'  , '0040', '0'),
         ('module/ddr'  , '0041', '2'),
         ('module/cpu'  , '0041', '1'),
         ('module/soc'  , '0040', '1'),
         ('module/wifi' , '0040', '2'),
 
         ('board/main'        , '0042', '0'),
         ('board/5v0-io-sys'  , '0042', '1'),
         ('board/3v3-sys'     , '0042', '2'),
         ('board/3v3-io-sleep', '0043', '0'),
         ('board/1v8-io'      , '0043', '1'),
         ('board/3v3-m.2'     , '0043', '2'),
         ],
    'jetson_nx':[('main', '0040','1'),
        ('cpu+gpu', '0040','2'),
        ('soc', '0040','3')
         ]
         }
driver_dir = {  'jetson_tx2':'/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/',
                'jetson_nx': '/sys/bus/i2c/devices/7-0040/'
            }

import os
# from jtop_logger import JtopLogger
# descr, i2c-addr, channel
# _nodes = [('module/main' , '0041', '0'),
#          ('module/gpu'  , '0040', '0'),
#          ('module/ddr'  , '0041', '2'),
#          ('module/cpu'  , '0041', '1'),
#          ('module/soc'  , '0040', '1'),
#          ('module/wifi' , '0040', '2'),
 
#          ('board/main'        , '0042', '0'),
#          ('board/5v0-io-sys'  , '0042', '1'),
#          ('board/3v3-sys'     , '0042', '2'),
#          ('board/3v3-io-sleep', '0043', '0'),
#          ('board/1v8-io'      , '0043', '1'),
#          ('board/3v3-m.2'     , '0043', '2'),
#          ]
 
_valTypes = ['power', 'voltage', 'current']
_valTypesFull = ['power [mW]', 'voltage [mV]', 'current [mA]']
 
def getNodes(device='jetson_tx2'):
    """Returns a list of all power measurement nodes, each a 
    tuple of format (name, i2d-addr, channel)"""
    assert device in device_nodes
    return device_nodes[device]
def getNodesByName(nameList=['module/main']):
    return [_nodes[[n[0] for n in _nodes].index(name)] for name in nameList]
 
def getDevice():
    for dir in driver_dir:
        if powerSensorsPresent(dir): 
            return dir 
def powerSensorsPresent(device='jetson_tx2'):
    """Check whether we are on the TX2 platform/whether the sensors are present"""
    return os.path.isdir(driver_dir[device])
 
def getPowerMode():
    return os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]
 
def readValue(i2cAddr='0041', channel='0', valType='power',device='jetson_tx2'):
    """Reads a single value from the sensor"""
    if device == 'jetson_tx2':
        fname = '/sys/bus/i2c/drivers/ina3221x/0-%s/iio:device%s/in_%s%s_input' % (i2cAddr, i2cAddr[-1], valType, channel)
        f = open(fname, 'r')
        res = f.read()
        f.close()
        return res  
    elif device == 'jetson_nx':
        res = {}
        for valtype in ['voltage','current']:
            val = 'in' if valtype=='voltage' else 'curr'
            # fname = '/sys/class/hwmon/hwmon4/%s%s_input' % (val,channel)
            fname='/sys/bus/i2c/drivers/ina3221/7-0040/hwmon/hwmon4/%s%s_input'% (val,channel)
            f = open(fname, 'r')
            res[valtype]  = f.read()
            f.close()
        res['power'] = eval(res['voltage'])*eval(res['current'])/1000
        return res[valType]
 
def getModulePower():
    """Returns the current power consumption of the entire module in mW."""
    return float(readValue(i2cAddr='0041', channel='0', valType='power'))
 
def getAllValues(nodes,device):
    """Returns all values (power, voltage, current) for a specific set of nodes."""
    return [[float(readValue(i2cAddr=node[1], channel=node[2], valType=valType,device=device)) for valType in _valTypes] for node in nodes]
 
def printFullReport(device):
    """Prints a full report, i.e. (power,voltage,current) for all measurement nodes."""
    # from tabulate import tabulate
    header = []
    header.append('A description')
    for vt in _valTypesFull:
        header.append(vt)
 
    resultTable = []
    for descr, i2dAddr, channel in device_nodes[device]:
        row = []
        row.append(descr)
        for valType in _valTypes:
            row.append(readValue(i2cAddr=i2dAddr, channel=channel, valType=valType,device=device))
        resultTable.append(row)
    
    import pandas as pd

    total = {}
    for i, vt in enumerate(header):
        if i<2: total.update({vt:[row[i] for row in resultTable]})
        else: total.update({vt:[eval(row[i]) for row in resultTable]})

    totalTable = pd.DataFrame(total)
    
    print(totalTable) # print(tabulate(resultTable, header))

"""
def draw_csv_img(csv_path = '/home/omnia/power/power/power.csv', names = list(range(12)), filename = 'from_csv', Events=None):
    from csv import reader
    label = []
    add_info = []
    data = []
    with open(csv_path,'r') as f:
        fcsv = reader(f)
        for row in fcsv:
            line = list(row)
            data.append(eval(line[2:]))
            label.append(line[1])
            add_info.append(eval(line[0]))
    
    x = data[0]
    y = data[1:]
    for i in range(1,len(data)):
        y.append(list(map(float,data[i])))
    
    import numpy as np
    x_s = []
    y_a = [[] for i in range(len(y))]
    y_t = [[] for i in range(len(y))]
    
    for i in range(int(len(x)/6)):
        x_s.append(i+3)
        for j in range(len(y)):
            part = y[j][i:i+6]
            y_a[j].append(np.mean(part))
            y_t[j].append(np.max(part))
            
    if names == None: 
        names = [name for name, _, _ in [self._nodes[i] for i in [0,1,2,3,11]]]
        label_names = ['System-wise', 'GPU', 'RAM', 'CPU', 'SSD']
        styles = [':', '-', '-', '-', '-', '--']
    else: 
        names = [name for name, _, _ in [self._nodes[i] for i in names]]
        label_names = names
        half = int((len(names)-1)/2)
        styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
    label_names.append('SUM')
                
    import matplotlib.pyplot as plt
    # total
    for t in range(len(y)): plt.plot(x, y[t], label = ('%s (%.2f J)' % (label[t+1], (add_info[t+1]/1e3))), linestyle=styles[t])
    plt.xlabel('time [s]')
    plt.ylabel(_valTypesFull[_valTypes.index(valType)])
    plt.grid(True)
    ln = plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) # ['%s (%.2f J)' % (name, enrgy/1e3) for name, enrgy in zip(label_names, energies)], 
    plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))
    if Events is not None:
        for t, _ in Events:
            plt.axvline(x=t, color='black', linestyle='-.')
            
    plt.savefig('power/%s/%s_total.png' % (filename, valType), bbox_extra_artists=(ln,), bbox_inches='tight')
    plt.close()
"""

import threading
import time

class PowerLogger:
    """This is an asynchronous power logger. 
    Logging can be controlled using start(), stop(). 
    Special events can be marked using recordEvent(). 
    Results can be accessed through 
    """
    # def __init__(self, interval=0.01, nodes=device_nodes['jetson_tx2'], figsave=True, csvwrite = True):
    def __init__(self, interval=0.01,figsave=True, csvwrite = True):
        """Constructs the power logger and sets a sampling interval (default: 0.01s) 
        and fixes which nodes are sampled (default: all of them)"""
        self.interval = interval
        self._startTime = -1
        self.eventLog = []
        self.dataLog = []
        self.figsave = figsave
        self.csvwrite = csvwrite
        self.device = getDevice()
        self._nodes = device_nodes[self.device]

    def start(self):
        "Starts the logging activity"""
        #define the inner function called regularly by the thread to log the data
        def threadFun():
            #start next timer
            self.start()
            #log data
            t = self._getTime() - self._startTime
            self.dataLog.append((t, getAllValues(self._nodes,self.device)))
            #ensure long enough sampling interval
            t2 = self._getTime() - self._startTime
            # assert(t2-t < self.interval)
             
        #setup the timer and launch it
        self._tmr = threading.Timer(self.interval, threadFun)
        self._tmr.start()
        if self._startTime < 0:
            self._startTime = self._getTime()
 
    def _getTime(self):
        return time.clock_gettime(time.CLOCK_REALTIME)
 
    def recordEvent(self, name):
        """Records a marker a specific event (with name)"""
        t = self._getTime() - self._startTime
        self.eventLog.append((t, name))
 
    def stop(self):
        """Stops the logging activity"""
        self._tmr.cancel()
 
    def getDataTrace(self, nodeName='module/main', valType='power'):
        # if getDevice() == 'jetson_nx': nodeName = 'main'
        """Return a list of sample values and time stamps for a specific measurement node and type"""
        pwrVals = [itm[1][[n[0] for n in self._nodes].index(nodeName)][_valTypes.index(valType)] 
                    for itm in self.dataLog]
        timeVals = [itm[0] for itm in self.dataLog]
        return timeVals, pwrVals
 
    def showDataTraces(self, names=None,valType='power', showEvents=True, filename='test'):
        """creates a PyPlot figure showing all the measured power traces and event markers"""
       
        device = getDevice()
        if device == 'jetson_tx2':
            if names == None: 
                names = [name for name, _, _ in [self._nodes[i] for i in [0,1,2,3,11]]]
                label_names = ['System-wise', 'GPU', 'RAM', 'CPU', 'SSD']
                styles = [':', '-', '-', '-', '-', '--']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
        elif device == 'jetson_nx':
            if names == None: 
                names = [name for name, _, _ in [node for node in self._nodes]]
                label_names = ['System-wise', 'GPU+CPU+CV', 'SOC']
                styles = [':', '-', '--']
            else: 
                names = [name for name, _, _ in [self._nodes[i] for i in names]]
                label_names = names
                half = int((len(names)-1)/2)
                styles = [":"] + ['-'] * half + ['-.'] * (len(names) - half) + ['--']
        #prepare data to display
        TPs = [self.getDataTrace(nodeName=name, valType=valType) for name in names]
        Ts, _ = TPs[0]

        Ps = [p for _, p in TPs]
        
        if device == 'jetson_tx2':
            Ps.append([Ps[1][i]+Ps[2][i]+Ps[3][i] for i in range(len(Ts))])
        elif device == 'jetson_nx':
            Ps.append([Ps[0][i]+Ps[1][i]+Ps[2][i] for i in range(len(Ts))])
        import numpy as np
        Ts_s = []
        Ps_a = []
        Ps_t = []
        for i in range(int(len(Ts)/5)):
            Ts_s.append(Ts[(5*i)+2])
        for p in range(len(Ps)):
            Ps_ap = []
            Ps_tp = []
            for i in range(int(len(Ts)/5)):
                part = Ps[p][(5*i):(5*i)+5]
                Ps_ap.append(np.mean(part))
                Ps_tp.append(np.max(part))
            Ps_a.append(Ps_ap)
            Ps_t.append(Ps_tp)
        
        energies = [self.getTotalEnergy(nodeName=nodeName) for nodeName in names]
        if device == 'jetson_tx2':
            energies.append(energies[1]+energies[2]+energies[3])
        elif device == 'jetson_nx':
            energies.append(energies[0]+energies[1]+energies[2])
 
        
        # Ps = list(map(list, zip(*Ps))) # transpose list of lists
        
        # label_names.append('SUM')
        os.makedirs('results_test/power/%s' % (filename,) , exist_ok=True)

        #draw figure
        if self.figsave:
            import matplotlib.pyplot as plt
            # total
            for t in range(len(label_names)): plt.plot(Ts, Ps[t], label = ('%s (%.2f J)' % (label_names[t], (energies[t]/1e3))), linestyle=styles[t])
            plt.xlabel('time [s]')
            plt.ylabel(_valTypesFull[_valTypes.index(valType)])
            plt.grid(True)
            ln = plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) # ['%s (%.2f J)' % (name, enrgy/1e3) for name, enrgy in zip(label_names, energies)], 
            plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))
            if showEvents:
                for t, _ in self.eventLog:
                    plt.axvline(x=t, color='black', linestyle='-.')
            
            plt.savefig('results_test/power/%s/%s_total.png' % (filename, valType), bbox_extra_artists=(ln,), bbox_inches='tight')
            plt.close()
            # Average
            for t in range(len(label_names)): plt.plot(Ts_s, Ps_a[t], label = ('%s (%.2f J)' % (label_names[t], (energies[t]/1e3))), linestyle=styles[t])
            plt.xlabel('time [s]')
            plt.ylabel(_valTypesFull[_valTypes.index(valType)])
            plt.grid(True)
            ln = plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) # ['%s (%.2f J)' % (name, enrgy/1e3) for name, enrgy in zip(label_names, energies)], 
            plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))
            if showEvents:
                for t, _ in self.eventLog:
                    plt.axvline(x=t, color='black', linestyle='-.')
            
            plt.savefig('results_test/power/%s/%s_average.png' % (filename, valType), bbox_extra_artists=(ln,), bbox_inches='tight')
            plt.close()
            # Top
            for t in range(len(label_names)): plt.plot(Ts_s, Ps_t[t], label = ('%s (%.2f J)' % (label_names[t], (energies[t]/1e3))), linestyle=styles[t])
            plt.xlabel('time [s]')
            plt.ylabel(_valTypesFull[_valTypes.index(valType)])
            plt.grid(True)
            ln = plt.legend(loc='center left', bbox_to_anchor=(1.04,0.5)) # ['%s (%.2f J)' % (name, enrgy/1e3) for name, enrgy in zip(label_names, energies)], 
            plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))
            if showEvents:
                for t, _ in self.eventLog:
                    plt.axvline(x=t, color='black', linestyle='-.')
            
            plt.savefig('results_test/power/%s/%s_top.png' % (filename, valType), bbox_extra_artists=(ln,), bbox_inches='tight')
            plt.close()
        
        if self.csvwrite: 
            import csv
            with open('results_test/power/%s/%s.csv' % (filename, valType),'w') as f:
                csvf = csv.writer(f)
                csvf.writerow(['time'] + label_names)
                for i in range(len(Ts)):
                    csvf.writerow([Ts[i]] + [Ps[j][i] for j in range(len(label_names))])
                csvf.writerow([0] + energies)

    def showMostCommonPowerValue(self, nodeName='module/main', valType='power', numBins=100, filename='test'):
        """computes a histogram of power values and print most frequent bin"""
        import numpy as np
        _, pwrData = np.array(self.getDataTrace(nodeName=nodeName, valType=valType))
        count, center = np.histogram(pwrData, bins=1) #int(len(pwrData)*5))
        if self.figsave:
            import matplotlib.pyplot as plt
            plt.bar((center[:-1]+center[1:])/2.0, count, align='center')
            plt.ylabel('Number of value')
            plt.xlabel(_valTypesFull[_valTypes.index(valType)])
            plt.grid(True)
            plt.title('%s trace (NVPModel: %s)' % (valType, os.popen("nvpmodel -q | grep 'Power Mode'").read()[15:-1]))
            os.makedirs('power/%s' % (filename,) , exist_ok=True)
            plt.savefig('power/%s/%s_histogram.png' % (filename, valType))
            plt.close()
        
        maxProbVal = center[np.argmax(count)]#0.5*(center[np.argmax(count)] + center[np.argmax(count)+1])
        print('max frequent %s bin value %s: %f' % (valType, (_valTypesFull[_valTypes.index(valType)])[-4:], maxProbVal))
        
        if self.csvwrite:
            import csv
            with open('power/%s/%s_histogram.csv' % (filename, valType),'w') as f:
                csvf = csv.writer(f)
                csvf.writerow(['x axis (center)', 'y axis (count)'])
                for i in range(len(count)):
                    csvf.writerow([center[i], count[i]])
 
    def getTotalEnergy(self, nodeName='module/main', valType='power'):
        """Integrate the power consumption over time."""
        timeVals, dataVals = self.getDataTrace(nodeName=nodeName, valType=valType)
        assert(len(timeVals) == len(dataVals))
        tPrev, wgtdSum = 0.0, 0.0
        for t, d in zip(timeVals, dataVals):
            wgtdSum += d*(t-tPrev)
            tPrev = t
        return wgtdSum
     
    def getAveragePower(self, nodeName='module/main', valType='power'):
        energy = self.getTotalEnergy(nodeName=nodeName, valType=valType)
        timeVals, _ = self.getDataTrace(nodeName=nodeName, valType=valType)
        return energy/timeVals[-1]


if __name__ == "__main__":

    printFullReport(getDevice())
    # print(getModulePower())
    # pl = PowerLogger(interval=0.05, nodes=getNodesByName(['module/main', 'board/main']))
    pl = PowerLogger(interval=0.05)
    pl.start()
    time.sleep(5)
    print('5s IDLE time passed, start IO bench mark now!')
    pl.recordEvent('started IO bench mark')
    time.sleep(2)
    pl.recordEvent('ding! 3s')
    os.system('stress -c 12 -t 3')
    time.sleep(1.5)
    pl.recordEvent('ding! 2s')
    os.system('stress -c 1 -t 2')
    time.sleep(2)
    pl.recordEvent('ding! 1s')
    os.system('stress -c 2 -t 1')
    time.sleep(1.5)
    
    pl.stop()
    pl.showDataTraces()
    # pl.showDataTraces(valType='voltage')
    # pl.showDataTraces(valType='current')
    nodename = device_nodes[pl.device][0][0] # main 
    pl.showMostCommonPowerValue(nodename)
    # pl.showMostCommonPowerValue(valType='voltage')
    # pl.showMostCommonPowerValue(valType='current')
