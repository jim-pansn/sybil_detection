import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NetworkFeatureGenerator():
    '''
        A class that can generate various kind of edge-features and apply them to a network
    '''

    def add_random_features(self, network, dim=10, name='random'):
        # list of all added feature names
        feature_list = []
        g = network.graph.structure    
        for i in range(dim):
            feature_list.append(name + str(i))
            for u, v, in g.edges():          
                g[u][v][name + str(i)] = np.random.random_sample()            
        return feature_list   


    def add_label_as_feature(self, network, name='label'):
        # list of all added feature names
        feature_list = [name]
        g = network.graph.structure    
        for u, v, in g.edges():          
            if (u, v) in network.attack_edges:
                g[u][v][name] = 1
            else:    
                g[u][v][name] = 0
        return feature_list  


    def add_edge_correlated_feature(self, network, name="correlated_feature", dim=8, iterations=1, stay_prob=0, stay_factor=1, norm=2, plot=False):
     
        # list of all added feature names
        feature_list = []

        # some useful information to buffer
        g = network.graph.structure
        order = g.order()
        honest_nodes = range(network.left_region.graph.order())
        sybil_nodes = range(network.left_region.graph.order(), order)
        label = np.ones(order)
        label[honest_nodes] = 0
        
        adja_mat = nx.to_numpy_matrix(g)
        # remove attacking edges to make sure there is no 
        # feature correlation between these nodes
        split_adja_mat = np.copy(adja_mat)
        split_adja_mat[np.ix_(honest_nodes, sybil_nodes)] = 0
        split_adja_mat[np.ix_(sybil_nodes, honest_nodes)] = 0

        # normalize the adjacency matrix and make it the transition matrix
        transition_matrix = split_adja_mat + np.eye(order) * stay_factor
        normalization = transition_matrix.sum(axis=1)[:, np.newaxis]  
        normalization[normalization == 0] = 1 # avoid dividing by zero
        transition_matrix = (transition_matrix / normalization)
        
        # add staying probability
        transition_matrix = transition_matrix * (1 - stay_prob) + np.eye(order) * stay_prob
        
        # generate initial feature distribution as a multivariate normal distribution
        mean = np.zeros(dim)
        cov = np.eye(dim)
        feature = np.random.multivariate_normal(mean, cov, order)
        
        if plot:
            # plot distribution of feature distances before the correlation process
            edge_sample = np.transpose(np.nonzero(split_adja_mat))
            no_edge_sample = np.transpose(np.nonzero(split_adja_mat == 0))
            edge_distances = np.linalg.norm(feature[edge_sample[:, 0]] - feature[edge_sample[:, 1]], axis=1)
            no_edge_distances = np.linalg.norm(feature[no_edge_sample[:, 0]] - feature[no_edge_sample[:, 1]], axis=1)    

            n, bins, patches = plt.hist([edge_distances, no_edge_distances])
            plt.show()
            dist1 = n[0] / float(n[0].sum())
            dist2 = n[1] / float(n[1].sum())
            plt.plot(dist1, label='edge')
            plt.plot(dist2, label='no edge')
            plt.legend()
            plt.show()

        # do the correlation / random walk
        for i in range(iterations):
            for d in range(dim):
                feature[:, d] = feature[:, d].dot(transition_matrix)
        if plot:
            # plot the distribution of the feature distances after the correlation process
            edge_distances = np.linalg.norm(feature[edge_sample[:, 0]] - feature[edge_sample[:, 1]], axis=1)
            no_edge_distances = np.linalg.norm(feature[no_edge_sample[:, 0]] - feature[no_edge_sample[:, 1]], axis=1)    

            n, bins, patches = plt.hist([edge_distances, no_edge_distances])
            plt.show()
            dist1 = n[0] / float(n[0].sum())
            dist2 = n[1] / float(n[1].sum())
            plt.plot(dist1, label='edge')
            plt.plot(dist2, label='no edge')
            plt.legend()
            plt.show()
        
        
        # apply the features to the network graph
        for d in range(dim):
            feature_list.append(name + str(d) + 'start')
            feature_list.append(name + str(d) + 'end')
            for u, v in g.edges():
                g[u][v][name + str(d) + 'start'] = feature[u, d]
                g[u][v][name + str(d) + 'end'] = feature[v, d]

        return feature_list



    