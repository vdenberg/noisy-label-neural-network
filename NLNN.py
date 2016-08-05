__author__ = 'esthervandenberg'

"""
This class NoisyLabelNeuralNetwork is initialized with a dataset and a model path. "run_NLNN"
then runs the EM module and updates the Neural Network for a set nr of iterations, saving under /Models each time
"""

from NN import NeuralNetwork
from EM import EMModule
import utils as ut

class NoisyLabelNeuralNetwork:
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None):
        self.NN = NeuralNetwork(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    def set_model_path(self, model_path):
        self.NN.model_path = model_path

    def pretrain(self, labels=None, noise_level=None, save_state=None, batch_size=100, epoch_size=10):
        # direct to NN function
        self.NN.train_NN(self, labels=labels, noise_level=noise_level, save_state=save_state, batch_size=batch_size, epoch_size=epoch_size)

    def run_NLNN(self, labels=None, noise_level=None, it_nr=15, batch_size=100, epoch_size=10):
        # set number of iterations of NLNN, 'before' model must already exist

        # retrieve predictions and initializes EM module, prints accuracy for information
        prob_y = self.NN.restored_prob_y(noise_level, state='before')
        self.NN.get_acc(noise_level, 'before')
        self.EM = EMModule(initializer=prob_y, labels=labels)

        # iterate, gets improved theta, updates NN and checks for convergence
        for it in range(it_nr):
            prev_theta = self.EM.theta
            c, new_theta = self.EM.iteration(it_nr=it, new_prob_y=prob_y)

            acc, prob_y = self.NN.train_NN(save_state='after', labels=c, noise_level=noise_level, batch_size=batch_size, epoch_size=epoch_size)

            if ut.dist(prev_theta, new_theta) < 10**-3:
                print('Converged after %s iterations\n'%it_nr)
                break

        # print accuracy for information
        self.NN.get_acc(noise_level, state='after')