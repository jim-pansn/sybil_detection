import networkx as nx
import numpy as np

class SybilRanker():
    '''
    Implementation of the SybilRank Algorithm
    '''

    def __init__(self, graph, start_nodes, num_iterations=30, deg_norm=True, total_ranking=1):
        self.graph = graph
        self.start_nodes = start_nodes
        self.num_iterations = num_iterations
        self.deg_norm = deg_norm
        self.total_ranking = total_ranking

        self.num_nodes = self.graph.order()

        self.adjacency_matrix = np.copy(nx.to_numpy_matrix(self.graph))
        self.transition_matrix = self.__get_transition_matrix()

        if self.deg_norm:
            deg_dict = self.graph.degree()
            self.deg = np.fromiter(iter(deg_dict.values()), dtype=float)
            self.norm_deg = self.deg / self.deg.sum()

    def __get_transition_matrix(self):
        normalization = self.adjacency_matrix.sum(axis=1)
        normalization[normalization == 0] = 1
        return self.adjacency_matrix / normalization[:, np.newaxis]    

    def rank(self):
        self.ranking = np.zeros(self.num_nodes)
        self.ranking[self.start_nodes] = float(self.total_ranking) / len(self.start_nodes)
        for i in range(self.num_iterations):
            self.ranking = self.ranking.dot(self.transition_matrix)
        if self.deg_norm:
             self.ranking = self.ranking / self.norm_deg
        return self.ranking
