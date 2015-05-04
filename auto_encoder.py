__author__ = 'teddy'

"""
This is based on the papers:
    A. Lemme, R. F. Reinhart and J. J. Steil.
    "Online learning and generalization of parts-based image representations
     by Non-Negative Sparse auto_encoders". Submitted to Neural Networks,
                              And
    A. Lemme, R. F. Reinhart and J. J. Steil. "Efficient online learning of
    a non-negative sparse auto_encoder". In Proc. ESANN, 2010.
"""
import numpy as np
import numpy.random as rand


class NNSAE:

    def __init__(self, inp_dim, hid_dim):
        self.inp_dim = inp_dim  # number of input neurons (and output neurons)
        self.hid_dim = hid_dim  # number of hidden neurons
        self.inp = np.zeros((inp_dim, 1))  # vector holding current input
        self.out = np.zeros((inp_dim, 1))  # output neurons
        self.g = np.zeros((inp_dim, 1))  # neural activity before non-linearity
        self.a = np.ones((hid_dim, 1))
        self.h = np.zeros((inp_dim, 1))  # hidden neuron activation
        self.b = -3 * np.ones((hid_dim, 1))
        # learning rate for synaptic plasticity of read-out layer (RO)
        self.lrateRO = 0.01
        self.regRO = 0.0002
        self.decayP = 0  # decay factor for positive weights [0..1]
        self.decayN = 1  # decay factor for negative weights [0..1]
        self.lrateIP = 0.001  # learning rate for intrinsic plasticity (IP)
        self.meanIP = 0.2  # desired mean activity, a parameter of IP
        self.W = 0.025 * \
            (2 * rand.rand(inp_dim, hid_dim) - 0.5 *
             np.ones((inp_dim, hid_dim))) + 0.025

    def apply(self, X):
        X = np.asmatrix(X)
        X = np.asarray(X.reshape(np.size(X),1))
        if np.asarray(X).shape[1] != 1:
            print('Use input which are Row Vectors of shape (1xL)')
            # exit(1)
        else:
            self.inp = X
            self.update()
            Xhat = self.out
            return Xhat

    def train(self, X):
        X = np.asmatrix(X)
        X = np.asarray(X.reshape(np.size(X), 1))
        if np.asarray(self.inp).shape[1] != 1:
            print('Use Inputs which are Row Vectors of shape (1xL)')
        # do forward propagation of activities
        self.update()

        # calculate adaptive learning rate
        lrate = self.lrateRO / (self.regRO + sum(np.power(self.h, 2)))

        # calculate erros
        error = self.inp - self.out

        # update weights
        self.W = self.W + lrate * error * self.h.transpose()

        # decay function for positive weights
        if self.decayP > 0:
            idx = np.where(self.W > 0)
            idx0 = list(idx[0])
            idx1 = list(idx[1])
            Len = len(idx[0])
            if Len > 0:
                for I in range(Len):
                    # net.W(idx) = net.W(idx) - net.decayP * net.W(idx)
                    self.W[idx0[I]][idx1[I]] = self.W[idx0[I]][
                        idx1[I]] - self.decayP * self.W[idx0[I]][idx1[I]]
        # decay functions for negative weights
        if self.decayN == 1:
            self.W = np.maximum(self.W, 0)
        elif self.decayN > 0:
            idx = np.where(self.W < 0)
            idx0 = list(idx[0])
            idx1 = list(idx[1])
            Len = len(idx[0])
            for I in range(Len):
                self.W[idx0[I]][idx1[I]] = self.W[idx0[I]][
                    idx1[I]] - self.decayN * self.W[idx0[I]][idx1[I]]
        else:
            pass
        # Intrinsic Plasticity
        hones = np.ones((self.hid_dim, 1))
        tmp = self.lrateIP * \
            (hones - (2.0 + float(1.0) / self.meanIP)
             * self.h + np.power(self.h, 2) / self.meanIP)
        self.b = self.b + tmp
        self.a = self.a + self.lrateIP * (hones / (self.a) + self.g * tmp)

    def update(self):
        self.g = np.dot(self.W.transpose(), self.inp)

        # Apply activation function
        self.h = float(1) / (1 + np.exp(-self.a * self.g - self.b))

        # Read-out
        self.out = np.dot(self.W, self.h)
