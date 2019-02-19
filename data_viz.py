import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import threading


def worker(i, message):
    print(i)
    print(message)
    print(i)

def get_dar_far(errors, labels):
    fpr, tpr, thresholds = roc_curve(labels, errors)
    max_score = 0
    optimal_index = np.argmax(np.abs(tpr-fpr))
    thresh = thresholds[optimal_index]
    predictions = np.zeros((len(errors),))
    for i in range(len(errors)):
        if errors[i] < thresh:
            predictions[i] = 0
        else:
            predictions[i] = 1
    total_attacks = 0
    attacks_correct = 0
    false_positive = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            attacks_correct +=1
        if labels[i] == 1:
            total_attacks +=1
        if predictions[i] ==1 and labels[i] ==0:
            false_positive +=1
            
    print("DAR")
    print(attacks_correct/total_attacks)
    print("FAR")
    print(false_positive/(len(errors)-total_attacks))
def get_data(FILE):
    return np.genfromtxt(FILE)



if __name__ == '__main__':
    '''
    threads= []
    for i in range(5):
        t = threading.Thread(target = worker, args = (i,"farts"))
        threads.append(t)
        t.start()
    exit()
    '''
    data = get_data('errors_tuesday_ensemble.csv')
    labels = get_data('tuesday_cl.csv')

   
    print(roc_auc_score(labels, data))
    get_dar_far(data, labels)
    exit()
    cd = []
    for i in range(len(labels)):
        if labels[i] == 0:
            cd.append('b')
        else:
            cd.append('r')
    time = [i for i in range(len(data))] 
    plt.scatter(time[0:30000], data[0:30000], c= cd[0:30000])
    plt.show()
    
