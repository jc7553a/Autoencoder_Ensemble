import numpy as np
import autoencoder as ae
import matplotlib.pyplot as plt
import random as ra
import CorrCluster as CC
import pandas as pd
import threading
import feature_methods as fm

error_layer = np.zeros((10,))
ensemble_layer = []
mapping = np.load('feature_mapping.npy') 
#mapping = []
def get_data(FILE):
    return np.genfromtxt(FILE)

def fix_v(v):
    buckets = []
    smalls = []
    for map in v:
        if len(map) <= 2:
            smalls.append(map)
            #pass
        else:
            buckets.append(map)
    smalls = [item for sublist in smalls for item in sublist]
    buckets.append(smalls)
    return buckets

def train_ensemble_batch(num, data):
    error_layer[num] = ensemble_layer[num].partial_fit(data[:][:,mapping[num]])

def train_ensemble(num, data):
    error_layer[num] = ensemble_layer[num].partial_fit([data[mapping[num]]])

def calc_ensemble(num, data):
    error_layer[num] = ensemble_layer[num].calc_total_cost([data[mapping[num]]])

def train_ensemble_batches(data, mapping, ensemble_layer, outer_layer):
    for i in range(5):
        print(i)
        for j in range(20000):
            rand = ra.randint(0,int(len(data))-5)
            threads = []
            for k in range(len(ensemble_layer)):
                t = threading.Thread(target = train_ensemble, args = (k, data[rand:rand+5]))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
    return ensemble_layer, outer_layer

def train(data, shape, mapping, ensemble_layer, outer_layer):
    for i in range(1):
        print(i)
        for j in range(0,len(data), 1):
            threads = []
            rand = ra.randint(0,len(data)-1)
            for k in range(len(ensemble_layer)):
                t = threading.Thread(target = train_ensemble, args = (k, data[rand]))
                threads.append(t)
                t.start()
            for thread in threads:
                thread.join()
            #for k in range(len(ensemble_layer)):
            #    error_layer[k] = ensemble_layer[k].partial_fit(data[[rand]][:,mapping[k]])
            outer_layer.partial_fit([error_layer])
    return ensemble_layer, outer_layer

def test(data, ensemble_layer, outer_layer, mapping):
    errors = np.zeros((len(data),))
    #for j in range(3):
    for i in range(len(data)):
        #for k in range(len(ensemble_layer)):
                #error_layer[k] = ensemble_layer[k].calc_total_cost(data[[i]][:,mapping[k]])
        threads = []
        for k in range(len(ensemble_layer)):
            t = threading.Thread(target = calc_ensemble, args = (k, data[i]))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()
        errors[i] = outer_layer.calc_total_cost([error_layer])
    return errors

def stupid():
    print(len(ensemble_errors))
    print(ensemble_errors)

if __name__ == '__main__':
    
    error_layer = np.zeros((len(mapping),))
    train_data = get_data('monday_reduced.csv')
    shape = np.shape(train_data)
   
    
    for a in mapping:
        ensemble_layer.append(ae.Autoencoder(len(a), int(len(a)*.75)))
    outer_layer = ae.Autoencoder(len(ensemble_layer), int(len(ensemble_layer)*.75))
    
    ensemble_layer, outer_layer = train_ensemble_batches(train_data, mapping, ensemble_layer, outer_layer)
    ensemble_layer, outer_layer = train(train_data, shape, mapping, ensemble_layer, outer_layer)
    del train_data
    print("Done Training")
    data2 = get_data('tuesday_reduced.csv')
    errors = test(data2, ensemble_layer, outer_layer, mapping)
    np.savetxt('errors_tuesday_ensemble.csv', errors)


    '''
    
     #Stuff might need to look at later
    #train_data = np.delete(train_data, [61, 60, 59, 58, 57, 56, 49, 33, 32, 31], axis = 1)
    #np.savetxt('tuesday_reduced.csv', train_data)
    #exit()
    #print(train_data[[0]][:,[42, 46, 65, 22, 27, 19, 16, 21, 24, 56, 40]])
    #exit()
    shape = np.shape(train_data)
    
    FM = CC.corClust(shape[1])
    for i in range(int(len(train_data)*.5)):
        FM.update(train_data[i])
    
    v = FM.cluster(5)
    v = fix_v(v)
    print(v)
    np.save('feature_mapping.npy', np.asarray(v))
    exit()    
    '''
