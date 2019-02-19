import autoencoder as ae
import numpy as np
import random as ra
from main import get_data

def train_batches(ann, data):
    for i in range(10):
        print(i)
        for j in range(2000):
            rand = ra.randint(0, int(len(data)*.5)-5)
            ann.partial_fit(data[rand:rand+5])
    return ann

def train(ann, data):
    for i in range(int(len(data)*.05), len(data),1):
        ann.partial_fit([data[i]])
    return ann

def test(ann, data):
    errors = np.zeros((len(data),))
    for i in range(len(data)):
        errors[i] = ann.calc_total_cost([data[i]])
    return errors

if __name__ == '__main__':
    data = get_data('monday_reduced.csv')
    ann = ae.Autoencoder(len(data[0]), len(data[0]*.75))
    ann = train_batches(ann, data)
    print("Done batches")
    ann = train(ann, data)
    print("Done training")
    del data
    data2 = np.genfromtxt('tuesday_reduced.csv')
    errors = test(ann, data2)
    np.savetxt('errors_single.csv', errors)
