import pandas as pd
import numpy as np

def corr_values(corr):
    size = int((len(corr[0])*(len(corr[0])+1))/(2)) - len(corr[0])
    corr_nums = np.zeros((size,))
    total = 0
    for i in range(1, len(corr[0])):
        j = 0
        while j < i :
            corr_nums[total] = corr[j][i]
            total+=1
            j +=1
    return corr_nums

def find_distances(value, nums):
    distances = np.zeros((len(nums),2 ))
    for i in range(len(nums)):
        distances[i, 0] = abs(value - nums[i])
        distances[i, 1] = i 
    return distances[distances[:,0].argsort()]

def split_spaces(nums, n):
    total = int(len(nums)/n)
    found = 0
    my_set = set()
    feature_map = []
    while found < total:
        got_it = False
        start = 0
        while not got_it:
            if start not in my_set:
                value = start
                got_it = True
            else:
                start += 1
        distances = find_distances(value, nums)
        temp_buckets = [value]
        my_set.add(value)
        start = 1
        while len(temp_buckets) < n:
            if distances[start,1] not in my_set:
                temp_buckets.append(distances[start, 1])
                my_set.add(distances[start,1])
                start+=1
            else:
                start +=1
        feature_map.append(temp_buckets)
        found +=1
    return feature_map

def feature_mapper(matrix, n):
    matrix = np.asarray(matrix)
    corr_vals = corr_values(matrix)
    return split_spaces(corr_vals,n) 
