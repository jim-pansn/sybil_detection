import networkx as nx
import numpy as np
from scipy.optimize import minimize

class SRWEarlyStopping():
    '''
    Implementation of the supervised Random walk with early stopping
    '''
    
    def __init__(self, graph, feature_list, start_nodes, positives, negatives,
                 strength_function='logistic', loss_function='hinge_loss', loss_factor=10,
                regularization_factor=1, iteration_stop=30, deg_norm=False, total_ranking=1):
        
        # given parameter
        self.graph = graph
        self.feature_list = feature_list
        self.start_nodes = start_nodes
        self.positives = positives
        self.negatives = negatives
        self.strength_function = strength_function
        self.loss_function = loss_function
        self.regularization_factor = regularization_factor
        self.loss_factor = loss_factor
        self.iteration_stop = iteration_stop
        self.total_ranking = total_ranking
        self.deg_norm = deg_norm

        # calculated and cached
        self.num_feature = len(self.feature_list)
        self.num_nodes = self.graph.order()

        self.adjacency_matrix = np.copy(nx.to_numpy_matrix(self.graph))
        self.feature_matrix = self.get_features_as_matrix()

        if self.deg_norm:
            deg_dict = self.graph.degree()
            self.deg = np.fromiter(iter(deg_dict.values()), dtype=float)
            self.norm_deg = self.deg / self.deg.sum()
        
    def get_features_as_matrix(self):
        feature_matrix = np.zeros((self.num_feature, self.num_nodes, self.num_nodes))
        for i, feature_name in enumerate(self.feature_list):
            feature_matrix[i] = nx.to_numpy_matrix(self.graph, weight=feature_name)
        return feature_matrix

    
    def rank(self):     
        self.ranking = np.zeros(self.num_nodes)
        self.ranking[self.start_nodes] = float(self.total_ranking) / len(self.start_nodes)
        converged = False
        i = 0
        while not converged:
            ranking_old = np.copy(self.ranking)
            self.ranking = self.ranking.dot(self.transition_matrix)
            i += 1
            if i > self.iteration_stop:
                break
            if(np.max(np.abs(self.ranking - ranking_old)) < .000001):
                converged = True
                print 'converged after', i, 'steps'

        if self.deg_norm:
            self.ranking = self.ranking / self.norm_deg  
        return self.ranking
        
        
    def update_weight(self, w):
        self.w = w
        self.strength_matrix = self.get_strength_matrix()
        self.transition_matrix = self.get_transition_matrix()       
    
    
    def get_transition_matrix(self):
        # normalize strength matrix
        normalization = self.strength_matrix.sum(axis=1)
        normalization[normalization == 0] = 1 # avoid dividing by zero
        q = (self.strength_matrix / normalization[:, np.newaxis])
        return q
    
    
    def get_strength_matrix(self):
        if(self.strength_function == 'logistic'):
            return self.get_logistic_strength_matrix()
        else:
            raise Exception('only logistic function implemented yet')         
    
        
    def get_logistic_strength_matrix(self):
        if np.any(-self.feature_matrix.T.dot(self.w) > 709):
            print 'ACHTUNG ACHTUNG OVERFLOW'
            print 'Exponent in logisitc_strngth_matrix:', -self.feature_matrix.T.dot(self.w)
            print 'w:', self.w
            print 'feature matrix:', self.feature_matrix
            raise Exception('Upcoming overflow in exp')
        result = 1 / (1 + np.exp(-self.feature_matrix.T.dot(self.w)))
        result[self.adjacency_matrix == 0] = 0
        return result
    
        
    def train(self):
        init_w = np.copy(self.w)
        res = minimize(self.calc_loss_with_weight, init_w)
        print 'res', res
        self.w = res.x
    
    def calc_loss(self):
        loss = self.calc_real_loss_without_reg()
        return (self.loss_factor * loss) + (np.linalg.norm(self.w)**2 * self.regularization_factor)
    
    
    def calc_loss_with_weight(self, weight):
        self.update_weight(weight)
        self.rank()
        return self.calc_loss()
    
    def calc_real_loss_without_reg(self):
        pairs = np.transpose([np.tile(self.positives, len(self.negatives)), np.repeat(self.negatives, len(self.positives))])
        b = .01
        loss = 0

        delta = self.ranking[pairs[:, 1]] - self.ranking[pairs[:, 0]]
        exponent = -delta / b
        if np.any(exponent > 709):
            print 'delta:', delta
            raise Exception('Upcoming overflow in exp')
        losses = 1 / (1 + np.exp(exponent))
        loss = losses.sum()
        return loss       


class SRWOneClass():
    '''
    One class SVM modelling of the Supervised Random Walk
    '''        

    def __init__(self, graph, feature_list, start_nodes, strength_function='logistic', loss_factor=10, regularization_factor=1, iteration_stop=30, deg_norm=False, total_ranking=1, nu=.2, rho_factor=10):
        
        # given parameter
        self.graph = graph
        self.feature_list = feature_list
        self.start_nodes = start_nodes
        self.strength_function = strength_function
        self.regularization_factor = regularization_factor
        self.loss_factor = loss_factor
        self.iteration_stop = iteration_stop
        self.total_ranking = total_ranking
        self.deg_norm = deg_norm
        self.nu = nu
        self.rho_factor = rho_factor

        # calculated and cached
        self.num_feature = len(self.feature_list)
        self.num_nodes = self.graph.order()

        self.adjacency_matrix = np.copy(nx.to_numpy_matrix(self.graph))
        self.feature_matrix = self.get_features_as_matrix()

        if self.deg_norm:
            deg_dict = self.graph.degree()
            self.deg = np.fromiter(iter(deg_dict.values()), dtype=float)
            self.norm_deg = self.deg / self.deg.sum()

    def get_features_as_matrix(self):
        feature_matrix = np.zeros((self.num_feature, self.num_nodes, self.num_nodes))
        for i, feature_name in enumerate(self.feature_list):
            feature_matrix[i] = nx.to_numpy_matrix(self.graph, weight=feature_name)
        return feature_matrix

    
    def rank(self):     
        self.ranking = np.zeros(self.num_nodes)
        self.ranking[self.start_nodes] = float(self.total_ranking) / len(self.start_nodes)
        converged = False
        i = 0
        while not converged:
            ranking_old = np.copy(self.ranking)
            self.ranking = self.ranking.dot(self.transition_matrix)
            i += 1
            if i > self.iteration_stop:
                break
            if(np.max(np.abs(self.ranking - ranking_old)) < .000001):
                converged = True
                print 'converged after', i, 'steps'

        if self.deg_norm:
            self.ranking = self.ranking / self.norm_deg  
        return self.ranking
        
        
    def update_weight(self, weight):
        self.w = weight[:-1]
        # self.threshold = weight[-2]
        self.rho = weight[-1]
        self.strength_matrix = self.get_strength_matrix()
        self.transition_matrix = self.get_transition_matrix()       
    
    
    def get_transition_matrix(self):
        # normalize strength matrix
        normalization = self.strength_matrix.sum(axis=1)
        normalization[normalization == 0] = 1 # avoid dividing by zero
        q = (self.strength_matrix / normalization[:, np.newaxis])
        return q
    
    
    def get_strength_matrix(self):
        if(self.strength_function == 'logistic'):
            return self.get_logistic_strength_matrix()
        else:
            raise Exception('only logistic function implemented yet')         
    
        
    def get_logistic_strength_matrix(self):
        if np.any(-self.feature_matrix.T.dot(self.w) > 709):
            print 'ACHTUNG ACHTUNG OVERFLOW'
            print 'Exponent in logisitc_strngth_matrix:', -self.feature_matrix.T.dot(self.w)
            print 'w:', self.w
            print 'feature matrix:', self.feature_matrix
            raise Exception('Upcoming overflow in exp')
        result = 1 / (1 + np.exp(-self.feature_matrix.T.dot(self.w)))
        #print 'result', result
        result[self.adjacency_matrix == 0] = 0
        return result
    
        
    def train(self):
        # init_w = np.append(self.w, [self.threshold, self.rho])
        init_w = np.append(self.w, self.rho)
        res = minimize(self.calc_loss_with_weight, init_w)
        print 'res', res
        self.w = res.x
    
    def calc_loss(self):
        loss = self.calc_real_loss_without_reg()
        
        return (self.loss_factor * loss) + (np.linalg.norm(self.w)**2 * self.regularization_factor)
    
    
    def calc_loss_with_weight(self, weight):
        # self.update_weight(weight)
        self.rho = weight[-1]
        self.rank()
        return self.calc_loss()
    
    def calc_real_loss_without_reg(self, details=False):
        xi_bound = self.rho - self.ranking
        #if details:
            #print 'before bound', xi_bound
        xi_bound [xi_bound < 0] = 0
        if details:
            #print 'after bound', xi_bound
            print 'sum', xi_bound.sum()
            print 'sum weighted:', (1. / (self.nu * self.num_nodes)) * xi_bound.sum()
        loss = (1. / (self.nu * self.num_nodes)) * xi_bound.sum() - (self.rho_factor * self.rho)
        return loss
        



class SRWSVM():
    '''

    '''
    
    def __init__(self, graph, feature_list, start_nodes, positives, negatives,
                 strength_function='logistic', loss_function='hinge_loss', loss_factor=10,
                regularization_factor=1, iteration_stop=30, deg_norm=False, total_ranking=1, c=1):
        
        # given parameter
        self.graph = graph
        self.feature_list = feature_list
        self.start_nodes = start_nodes
        self.positives = positives
        self.negatives = negatives
        self.strength_function = strength_function
        self.loss_function = loss_function
        self.regularization_factor = regularization_factor
        self.loss_factor = loss_factor
        self.iteration_stop = iteration_stop
        self.total_ranking = total_ranking
        self.deg_norm = deg_norm
        self.c = c

        # calculated and cached
        self.num_feature = len(self.feature_list)
        self.num_nodes = self.graph.order()

        self.adjacency_matrix = np.copy(nx.to_numpy_matrix(self.graph))
        self.feature_matrix = self.get_features_as_matrix()

        if self.deg_norm:
            deg_dict = self.graph.degree()
            self.deg = np.fromiter(iter(deg_dict.values()), dtype=float)
            self.norm_deg = self.deg / self.deg.sum()
        
    def get_features_as_matrix(self):
        feature_matrix = np.zeros((self.num_feature, self.num_nodes, self.num_nodes))
        for i, feature_name in enumerate(self.feature_list):
            feature_matrix[i] = nx.to_numpy_matrix(self.graph, weight=feature_name)
        return feature_matrix

    
    def rank(self):     
        self.ranking = np.zeros(self.num_nodes)
        self.ranking[self.start_nodes] = float(self.total_ranking) / len(self.start_nodes)
        converged = False
        i = 0
        while not converged:
            ranking_old = np.copy(self.ranking)
            self.ranking = self.ranking.dot(self.transition_matrix)
            i += 1
            if i > self.iteration_stop:
                break
            if(np.max(np.abs(self.ranking - ranking_old)) < .000001):
                converged = True
                print 'converged after', i, 'steps'

        if self.deg_norm:
            self.ranking = self.ranking / self.norm_deg  
        return self.ranking
        
        
    def update_weight(self, w):
        self.w = w[:-2]
        self.svm_w = w[-2]
        self.b = w[-1]
        self.strength_matrix = self.get_strength_matrix()
        self.transition_matrix = self.get_transition_matrix()       
    
    
    def get_transition_matrix(self):
        # normalize strength matrix
        normalization = self.strength_matrix.sum(axis=1)
        normalization[normalization == 0] = 1 # avoid dividing by zero
        q = (self.strength_matrix / normalization[:, np.newaxis])
        return q
    
    
    def get_strength_matrix(self):
        if(self.strength_function == 'logistic'):
            return self.get_logistic_strength_matrix()
        else:
            raise Exception('only logistic function implemented yet')         
    
        
    def get_logistic_strength_matrix(self):
        if np.any(-self.feature_matrix.T.dot(self.w) > 709):
            print 'ACHTUNG ACHTUNG OVERFLOW'
            print 'Exponent in logisitc_strngth_matrix:', -self.feature_matrix.T.dot(self.w)
            print 'w:', self.w
            print 'feature matrix:', self.feature_matrix
            raise Exception('Upcoming overflow in exp')
        result = 1 / (1 + np.exp(-self.feature_matrix.T.dot(self.w)))
        result[self.adjacency_matrix == 0] = 0
        return result
    
        
    def train(self):
        init_w = np.append(self.w, [self.svm_w, self.b])
        res = minimize(self.calc_loss_with_weight, init_w)
        print 'res', res
        self.w = res.x
    
    def calc_loss(self):
        loss = self.calc_real_loss_without_reg()
        return (self.loss_factor * loss) + (np.linalg.norm(self.w)**2 * self.regularization_factor)
    
    
    def calc_loss_with_weight(self, weight):
        self.update_weight(weight)
        #self.svm_w = weight[-2]
        #self.b = weight[-1]
        self.rank()
        return self.calc_loss()
    
    def calc_real_loss_without_reg(self):
        xi_bound_positives = 1 - (self.svm_w * self.ranking[self.positives] + self.b)
        xi_bound_positives[xi_bound_positives < 0] = 0
        loss = xi_bound_positives.sum()
        
        xi_bound_negatives = 1 + (self.svm_w * self.ranking[self.negatives] + self.b)
        xi_bound_negatives[xi_bound_negatives < 0] = 0
        loss += xi_bound_negatives.sum()
        
        loss = loss * self.c + np.abs(self.svm_w)
        print '\n TOTAL LOSS:', loss, '\n'
        return loss 




