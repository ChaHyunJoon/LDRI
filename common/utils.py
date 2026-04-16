import os
import sys
import numpy as np
import scipy.io as scio
import xlrd

def get_driving_cycle(cycle_name):
    cycle_dir = "/home/chahyunjoon/dev/FCEV-EMS/project-data/standard_driving_cycles/mat-data/"
    filename = cycle_dir + cycle_name + '.mat'     
    data = scio.loadmat(filename)
    speed_list = data['speed_vector'][0]            # class <ndarray>
    return speed_list

def get_driving_cycle_v2(cycle_name):
    cycle_dir = "/home/chahyunjoon/dev/FCEV-EMS/project-data/standard_driving_cycles/mat-data/"
    filename = cycle_dir + cycle_name
    data_sheet = xlrd.open_workbook(filename + '.xls')
    table = data_sheet.sheets()[0] 
    speed_list = table.col_values(0)  # class <list>        # speed list of leading car, be observed by rear car
    return speed_list

def get_acc_limit(speed_list, output_max_min=False):
    num = len(speed_list)
    acc_list = []
    for i in range(1, num):
        acc_list.append(speed_list[i]-speed_list[i-1])
    acc_list.append(0)
    if output_max_min:
        max_acc = max(acc_list)
        min_acc = min(acc_list)
        return acc_list, max_acc, min_acc
    else:
        return acc_list


def summarize_fc_efficiency(p_fc, fce_eff, on_threshold=0.0):
    p_fc_arr = np.asarray(p_fc, dtype=np.float64).reshape(-1)
    fce_eff_arr = np.asarray(fce_eff, dtype=np.float64).reshape(-1)

    if p_fc_arr.size == 0:
        return 0.0, 0.0

    fc_on_mask = np.isfinite(p_fc_arr) & (p_fc_arr >= float(on_threshold))
    pct_fc_on = float(np.mean(fc_on_mask)) * 100.0
    if not np.any(fc_on_mask):
        return 0.0, pct_fc_on

    fce_eff_on = fce_eff_arr[fc_on_mask]
    fce_eff_on = fce_eff_on[np.isfinite(fce_eff_on)]
    if fce_eff_on.size == 0:
        return 0.0, pct_fc_on

    return float(np.mean(fce_eff_on)), pct_fc_on

class Logger:
    """
    save log automaticly
    """
    
    def __init__(self, filepath, filename, stream=sys.stdout):
        self.terminal = stream
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # create a note.txt
        notename = filepath+'note.txt'
        if not os.path.exists(notename):
            file = open(filepath+'note.txt', 'w')
            file.write('-----Configuration note-----'+'\n')
            file.close()
        
        self.log = open(filepath+filename, 'a') 
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass
