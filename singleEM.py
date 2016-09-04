__author__ = 'esthervandenberg'

import argparse as ap
import utils as ut
from EM import EMModule
import numpy as np
import os
import ConLLtoVector as ctv

def get_xy_train(train_path):
    sent_end_marker = '\n'
    with open(train_path) as f:
        lines = f.readlines()
        x_uni = []
        y_uni = []
        for l in lines[2:]:
            if l != '\n':
                l = l.strip().split(' ')
                x_uni.append(l[0])
                y_uni.append(l[-1])
            else:
                x_uni.append(sent_end_marker)
                y_uni.append(sent_end_marker)
    return x_uni, y_uni

## not really needed anymore
def xy_train_to_noiselev(train_path):
    noise_level = train_path[5:][:-10]
    print('Noise level correct?:', noise_level)
    return noise_level

def save_xy_train(x, y, path):
    with open(modelpath, 'w') as f:
        for i in range(len(x)):
            f.write(x[i] + '\t' + y[i] + '\n')
    print('Saved to {}'.format(modelpath))
##

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Bilty (based on existing model) with EM', usage='singleEM [-noise] [-initialize]')

    parser.add_argument('-noise', help="Give noise level")
    parser.add_argument('-initialize', help="Model path", action="store_true")
    args = parser.parse_args()

    # what about these? --output predictions --pred_layer 1 --iters 30 --h_dim 50 --save Models/00before

    # retrieve predictions and initializes EM module
    estlab_path = 'NER_estimatedlabels/noisy' + args.noise + '_eng.train'
    estlab_data = ctv.toVector(estlab_path)
    noiselab_data = ctv.toVector('NER_uninoisy/noisy' + args.noise + '_eng.train')

    y_train = noiselab_data.y_vector
    prob_y = estlab_data.y_vector
    nr_epochs=30

    if args.initialize:
        EM = EMModule(initialize=True, initializer=prob_y, labels=y_train)
        np.save('theta', EM.theta)
    else:
        EM = EMModule(labels=y_train)
        EM.theta = np.load('theta.npy')
    prev_theta = EM.theta

    # iterates once, gets improved theta, updates and checks for convergence
    print("Updating theta and c")
    c, new_theta = EM.iteration(new_prob_y=y_train)

    c_data = ctv.toCoNLL(estlab_data.x_uni, c, estlab_data.labelset)
    c_data.extend_to_preds(noiselab_data.y_uni)
    c_data.save_ConLL_data(c_data.as_preds, estlab_path+'test')


