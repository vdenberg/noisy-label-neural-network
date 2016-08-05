__author__ = 'esthervandenberg'

import random
import numpy as np

def random_select(P):
    # takes a sequence P of real positive numbers,
    # randomly selects an element p and return its index i

    # construct scale of classes
    accum_P = [P[0]]
    for i in range(1, len(P)):
        accum_P.append(accum_P[i-1]+P[i])

    # guess a number and find associated class
    i = random.uniform(0.01, .99)

    find_interval = list(accum_P)
    find_interval.append(i)
    find_interval.sort()

    i_idx = find_interval.index(i)
    right_bound = find_interval[i_idx+1]

    label = accum_P.index(right_bound)

    """
    print(accum_P)
    print(i)
    print(find_interval)
    print(i_idx)
    print(right_bound) ###
    print(label)
    """

    return label

def make_uni_noisy(labels, p=0.1):
    nr_instances = labels.shape[0]
    nr_classes = labels.shape[1]

    # construct theta based on p
    zt = get_zt(labels)
    th = np.zeros([nr_classes]*2)
    for i in range(nr_classes):
        th[i] = [p/(nr_classes-1)]*10
        th[i,i] = 1-p/(nr_classes-1)*9

    # generate noisy labels using theta
    noisy_labels = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        P = th[zt[t]]
        label = random_select(P)
        noisy_labels[t, label] = 1

    return noisy_labels

def get_zt(labels):
    nr_instances = labels.shape[0]
    return [list(labels[t]).index(1) for t in range(nr_instances)]

def make_perm_noisy(labels, p=0.1):
    nr_instances = labels.shape[0]
    nr_classes = labels.shape[1]

    # construct theta based on p
    zt = get_zt(labels)
    th = np.zeros([nr_classes]*2)

    print(1-p/(nr_classes-1)*9)
    print(1-p/(nr_classes-1))

    for i in range(nr_classes):
        n_frac = float(p)
        for j in range(nr_classes):
            if not i == j:
                th[i,j] = random.uniform(0, 0.10)
                n_frac -= th[i,j]
            else:
                th[i,j] = 1-p

    print(th)

    # generate noisy labels using theta
    noisy_labels = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        P = th[zt[t]]
        label = random_select(P)
        noisy_labels[t, label] = 1

    return noisy_labels

def dist(A, B):
    return abs(get_frob_norm(A) - get_frob_norm(B))

def get_frob_norm(A):
    return np.sqrt(np.trace(np.dot(np.transpose(A), A)))

def get_reverse_zt(c):
    nr_instances = c.shape[0]
    nr_classes = c.shape[1]
    reverse_zt = np.zeros([nr_instances, nr_classes])
    for t in range(nr_instances):
        i = list(c[t]).index(max(c[t]))
        reverse_zt[t,i] = 1
    return reverse_zt

def get_batches(x, y, batch_size=100, epoch_size=10):
    nr_instances = x.shape[0]

    images = x[:]
    labels = y[:]

    batches_x = []
    batches_y = []

    for ep in range(epoch_size):
        start = 0
        while start < nr_instances:
            batches_x.append(images[start:start+batch_size])
            batches_y.append(labels[start:start+batch_size])
            start += batch_size

        perm = np.arange(np.shape(x)[0])
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]

    return batches_x, batches_y

def frac_from_lev(noise_level):
    frac = float(noise_level[0] + '.' + noise_level[1])
    return frac

# delete certain classes
def delete_classes(data_x, data_y, tbd_classes=[7,2,5,3]):

    confclass_y = data_y[:,tbd_classes]
    if confclass_y.shape[1] != len(tbd_classes):
        print('Too many columns')

    tbd_inds = []
    for ind in range(len(confclass_y)):
        if sum(confclass_y[ind]) != 0:
            tbd_inds.append(ind)

    #tbd_inds = [ind for ind in range(len(confclass_y)) if confclass_y[ind]*1 != 0]

    # delete tbd_inds from data_x & data_y
    inds = range(len(confclass_y))
    keep_inds = list(set(inds) - set(tbd_inds))

    data_x = data_x[keep_inds,:]
    data_y = data_y[keep_inds,:]

    print('New length: ', len(inds), len(keep_inds))
    return (np.asarray(data_x), np.asarray(data_y))

# play around with data size
def sample_data(data_x, data_y):
    nr_instances = data_x.shape[0]
    sample = random.sample(range(nr_instances), int(np.ceil(nr_instances*0.1)))

    print('New length: ', len(data_x), len(data_x[sample]))
    return (data_x[sample], data_y[sample])

def save_noisy(labels, noise_level):
    noisy = make_uni_noisy(labels, frac_from_lev(noise_level))
    np.save("Noisy_data/train-labels-idx1-ndarray" + noise_level, noisy)

def load_noisy(noise_level):
    return np.load("Noisy_data/train-labels-idx1-ndarray" + noise_level + ".npy")