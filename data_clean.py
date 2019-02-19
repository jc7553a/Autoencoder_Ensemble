import numpy as np
import pandas as pd

def normalize(data):
    maxs = np.genfromtxt('maxs.csv')
    mins = np.genfromtxt('mins.csv')
    for i in range(len(data[0])):
        mx = maxs[i]
        mi = mins[i]
        for j in range(len(data)):
            data[j][i] = (data[j][i]-mi)/(mx-mi)
            if np.isnan(data[j][i]):
                data[j][i] = 0
    return data
def fix_data(data):
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data[i][j] = np.float(data[i][j])
            if np.isnan(data[i][j]) or np.isinf(data[i][j]) or np.isnan(data[i][j]):
                data[i][j] = 0
    return data

def find_min_max(data):
    maxs = []
    mins = []
    for i in range(len(data[0])-1):
        maxs.append(np.amax(data[:,i]))
        mins.append(np.amin(data[:,i]))

    np.savetxt('maxs.csv', maxs)
    np.savetxt('mins.csv', mins)

def print_class_vals(data):
    f = open('tuesday_labels.csv','w')
    for i in range(len(data)-1):
        f.write(str(data[i]) +',')
    f.write(str(data[-1]))
    f.close()

if __name__ == '__main__':
    data = pd.read_csv("./MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=False)
    data = np.asarray(data)
    data = fix_data(data)
    clas_vals = data[:,-1]
    #find_min_max(data)
    print_class_vals(clas_vals)
    data = np.delete(data, -1, axis = 1)

    data = normalize(data)
    np.savetxt('tuesday.csv', data)
    np.savetxt('tuesday_classvals.csv', clas_vals)
