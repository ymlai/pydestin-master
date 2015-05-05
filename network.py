# -*- coding: utf-8 -*-
__author__ = 'teddy'
import scipy.io as io
from load_data import *
from layer import *

# io.savemat(file_name,Dict,True)
# TODO: get ridoff the sequential requirements like first feed the layer
# an input the you can initialize it


class Network():

    def __init__(self, num_layers, alg_choice, alg_params, num_nodes_per_layer, cifar_stat, patch_mode='Adjacent', image_type='Color'):
        self.network_belief = {}
        self.lowest_layer = 1
        # this is going to store beliefs for every image DeSTIN sees
        self.network_belief['belief'] = np.array([])
        self.save_belief_option = 'True'
        self.belief_file_name = 'beliefs.mat'
        self.number_of_layers = num_layers
        self.algorithm_choice = alg_choice
        self.algorithm_params = alg_params
        self.number_of_nodesPerLayer = num_nodes_per_layer
        self.patch_mode = patch_mode
        self.image_type = image_type
        self.layers = [
            [Layer(j, num_nodes_per_layer[j], cifar_stat, self.patch_mode, self.image_type) for j in range(num_layers)]]


    def setmode(self, mode):
        self.operating_mode = mode
        for I in range(self.number_of_layers):
            self.layers[0][I].mode = mode

    def init_network(self):
        for L in range(self.number_of_layers):
            self.initLayer(L)

    def set_lowest_layer(self, lowest_layer):
        self.lowest_layer = lowest_layer

    def initLayer(self, layer_num):
        self.layers[0][layer_num].init_layer_learning_params(
            self.algorithm_choice, self.algorithm_params)

    def train_layer(self, layer_num):
        self.layers[0][layer_num].do_layer_learning(self.operating_mode)

    def update_belief_exporter(self):
        for i in range(self.lowest_layer, self.number_of_layers):
            for j in range(len(self.layers[0][i].nodes)):
                for k in range(len(self.layers[0][i].nodes[0])):
                    if self.network_belief['belief'] == np.array([]):
                        self.network_belief['belief'] = np.array(
                            self.layers[0][i].nodes[j][k].belief).ravel()
                    else:
                        self.network_belief['belief'] = np.hstack((np.array(self.network_belief['belief']),
                                                                  np.array(self.layers[0][i].nodes[j][k].belief).ravel()))



    def update_pool_belief_exporter(self):
       for i in range(self.lowest_layer, self.number_of_layers):
        if (i==self.number_of_layers-1):         # Top layer no need to pool
           self.network_belief['belief'] = np.hstack((np.array(self.network_belief['belief']),np.array(self.layers[0][i].nodes[0][0].belief).ravel()))
        else:
            l=len(self.layers[0][i].nodes)
            self.pool_belief(0,l/2,0,l/2,i)          #pool belief vector from 1/4 layer
            self.pool_belief(l/2,l,0,l/2,i)
            self.pool_belief(0,l/2,l/2,l,i)
            self.pool_belief(l/2,l,l/2,l,i)
            #self.pool_belief(0,l,0,l,i)
            


    def pool_belief(self, row_start, row_end, col_start, col_end, layer_index):
        temp_network_belief=np.array([0])
        for j in range(row_start, row_end):
                for k in range(col_start, col_end):
                        temp_network_belief = np.array(self.layers[0][layer_index].nodes[j][k].belief).ravel()+temp_network_belief
        self.network_belief['belief'] = np.hstack((np.array(self.network_belief['belief']),temp_network_belief.ravel()))
        #print "temp_network_belief=np.array([0])",temp_network_belief
        #print "self.network_belief['belief']",self.network_belief['belief']
        #print "len",len(self.network_belief['belief'])
        #print "=====================================================+"





    def dump_belief(self, num_of_images):
        total_belief_len = len(np.array(self.network_belief).ravel())
        single_belief_len = total_belief_len / num_of_images
        #print np.array(self.network_belief).ravel()
        belief = np.array(self.network_belief).reshape(
            num_of_images, single_belief_len)
        io.savemat(self.belief_file_name, belief)

    def clean_belief_exporter(self):
        self.network_belief['belief'] = np.array([])

