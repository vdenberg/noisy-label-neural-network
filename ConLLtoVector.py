__author__ = 'esthervandenberg'

import os
import sys
import random

import numpy as np
import argparse as ap
"""
Goal: take CONLL data, inject noise, and save as noisy data
"""

def CoNLL_to_uni(data_path, sent_end_marker='\n'):
    with open(data_path) as f:
        lines = f.readlines()
        x_uni = []
        y_uni = []
        for l in lines[2:]:
            if l != '\n':
                l = l.strip().split('	')
                x_uni.append(l[0])
                y_uni.append(l[-1])
            else:
                x_uni.append(sent_end_marker)
                y_uni.append(sent_end_marker)
    return x_uni, y_uni

def uni_to_uni_num(uni, uni_set):
    inds = {uni_set[i]: i for i in range(len(uni_set))}
    return [inds[uni[i]] for i in range(len(uni))]

def uni_to_xy_uni(x, y):
    joiner = '\t'
    xy = zip(x,y)
    xy = [joiner.join(list(tup)) for tup in xy]
    xy = [line.replace('\n\t\n', '') for line in xy]
    return xy

def get_reverse_zt(c, start_from_0=True):
    nr_instances = len(c)
    nr_classes = max(c)+1
    reverse_zt = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        if start_from_0:
            labelind = c[t]
        else:
            labelind = c[t]-1
        reverse_zt[t,labelind] = 1
    return reverse_zt

class toVector():
    def __init__(self, path):
        self.x_uni, self.y_uni = CoNLL_to_uni(path, sent_end_marker = '\n')
        sent_end_marker='\n'
        self.labelset = list(set(self.y_uni))
        self.labelset.pop(self.labelset.index(sent_end_marker))
        self.labelset = self.labelset + [sent_end_marker]
        self.y_uni_num = uni_to_uni_num(self.y_uni, self.labelset)
        """
        print(path, 'x_uni', self.x_uni[:10])
        print(path, 'y_uni', self.y_uni[:10])
        print(path, 'y_uni_num', self.y_uni_num[:10])
        print(path, 'labelset', self.labelset)
        """

        self.label_size = max(self.y_uni_num)+1
        #print('Label size', self.label_size)

        self.y_vector = get_reverse_zt(self.y_uni_num)
        #print(path, 'y_vector', self.y_vector[:3])

def uni_num_to_uni(uni_num, labelset):
    uni = [labelset[n] for n in uni_num]
    return uni

def get_zt(labels):
    nr_instances = labels.shape[0]
    return [list(labels[t]).index(1) for t in range(nr_instances)]

class toCoNLL():
    def __init__(self, x_uni, noise_vector_labels, labelset):
        self.x_uni = x_uni
        self.y_uni_num = get_zt(noise_vector_labels)
        self.y_uni = uni_num_to_uni(self.y_uni_num, labelset)
        self.xy_uni = uni_to_xy_uni(x_uni, self.y_uni)

    def extend_to_preds(self, origs):
        self.as_preds = zip(self.x_uni, origs, self.y_uni)
        return self.as_preds

    def save_ConLL_data(self, conll_data, output_path):
        with open(output_path, 'w') as f:
            for i in range(conll_data):
                f.write('\t'.join(list(conll_data[i])) + '\n')

#for ps, ds, fs in os.walk('NER_estimatedlabels/'):
#    for f in fs:
#        data = toVector('NER_estimatedlabels/' + f) # convert ConLL to indices, and indices to vectors
#        vector_labels = data.y_vector # grab vector
#        np.save('NER_estimatedlabels/' + f + '.npy', vector_labels) # save vector
