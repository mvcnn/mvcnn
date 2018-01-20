import os
import csv
import pickle

'''
with open('C://Users/Aaron Chadha/Downloads/kinetics_train/kinetics_train/kinetics_train.csv') as f:
    reader = csv.reader(f)
    d = dict()
    i = 0
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        if row[0] not in d.keys():
            d[row[0]] = i
            i += 1
print(d.keys(), len(d))


with open('class_list.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=2)
'''

with open('class_list.pickle', 'rb') as handle:
    d = pickle.load(handle)

print(d['ice skating'])



