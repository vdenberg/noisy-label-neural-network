__author__ = 'esthervandenberg'

"""
Running this script will start an experiment in one of the following three modes:

-pretrain   trains neural networks on noisy data given a set of noise fractions
-EM         runs an EM module based on aforementioned models
-plot       plots results
"""

import argparse as ap
from NLNN import NoisyLabelNeuralNetwork
from tensorflow.examples.tutorials.mnist import input_data
import utils as ut
import os

if __name__ == "__main__":

    parser = ap.ArgumentParser(description='All you need for an NLNN experiment on MNIST data', usage='NLNNonMNIST [-h] [-pretrain] [-EM] [-plot]')

    parser.add_argument('-pretrain', help="Train NN models", action="store_true")
    parser.add_argument('-EM', help="Train NNLN models", action="store_true")
    parser.add_argument('-plot', help="Plot results of all available models", action="store_true")
    args = parser.parse_args()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    noise_levels = ['00','01','02','03','04','05','06']

    for lev in noise_levels:
        # save noisy data at each noise level
        ut.save_noisy(mnist.train.labels, lev)

    if args.pretrain:
        # trains model without EM module and save under /Models
        model = NoisyLabelNeuralNetwork(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)
        model.set_model_path("/Users/esthervandenberg/MA/UdS/Thesis/NoisyLabelNeuralNetwork/Models/")
        for lev in noise_levels:
            model.pretrain(labels=ut.load_noisy(lev), noise_level=lev, save_state='before', epoch_size=50)

    elif args.EM:
        # trains with EM module and save to under /Models
        modelEM = NoisyLabelNeuralNetwork(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)
        modelEM.set_model_path("/Users/esthervandenberg/MA/UdS/Thesis/NoisyLabelNeuralNetwork/Models/")
        for lev in noise_levels:
            modelEM.run_NLNN(labels=ut.load_noisy(lev), noise_level=lev, it_nr=15, epoch_size=50)

    elif args.plot:
        # summarizes/visualising accuracies of models in /Models
        print("Plotting function not finished, will list accuracies instead")
        for pth, dirs, files in os.walk('/Users/esthervandenberg/MA/UdS/Thesis/NoisyLabelNeuralNetwork/Models/'):
            for filename in files:
                if not filename.endswith('meta') and filename.startswith('0'):
                    state = filename[2:-5]
                    noise_level = filename[:2]
                    nn = NoisyLabelNeuralNetwork(test_x=mnist.test.images, test_y=mnist.test.labels)
                    nn.set_model_path('Models/')
                    acc = nn.NN.get_acc(noise_level, state)

    else:
        print("Please specify what the model should do")

