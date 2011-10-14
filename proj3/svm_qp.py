import random

from numpy import zeros,dot, vstack, hstack,identity, ones, array

from cvxopt import solvers, matrix

from load import load_project_data
        
class SVM(object): 
    
    def __init__(self,training_set,C):
        self.C = C
        self.train(training_set)
        
        
    def train(self,training_set):
    
        self.training_set = array(training_set.to_float()) 
    
        n_features = len(self.training_set[0][1:-1])
        n_training = len(self.training_set)
        
        n_x = 1+ n_features+ n_training
        
        #standard form for qp solver: 
        #(1/2)*x^T*P*x + q^T*x; G*x <= h
                                
        p_1 = zeros((n_x,1))
        p_21 = zeros((1,n_features))
        p_22 = identity(n_features)
        p_23 = zeros((n_training,n_features))
        p_2  = vstack((p_21,p_22,p_23))
        p_3  = zeros((n_x,n_training))
        P    = matrix(hstack((p_1,p_2,p_3))) #P
        
        q_1 = zeros((1,1+n_features))
        q_2 = self.C*ones((1,n_training))
        q   = matrix(hstack((q_1,q_2))).T #q
        
        y = array([self.training_set[:,-1]])
        y = (y==0).choose(y,-1) #replace all 0's with -1
        g_1 = y.T
        g_2 = hstack((y.T,)*n_features)*self.training_set[:,1:-1]
        g_3 = identity(n_training)
        g_top = hstack((g_1,g_2,g_3))
        g_4 = zeros((n_training,1+n_features))
        g_5 = identity(n_training)
        g_bottom = hstack((g_4,g_5))
        G = matrix(-1*vstack((g_top,g_bottom))) #G
        
        h_1 = -1*ones((n_training,1))
        h_2 = zeros((n_training,1))
        h = matrix(vstack((h_1,h_2))) #h
        
        sol=solvers.qp(P, q, G, h)
        
        print sol['x']
        
        
        
if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    C = float(sys.argv[2])
    
    random.seed(12345)
    
    folds = load_project_data(data_name,3)
    
    s = SVM(folds[0][0],C)        
        
        
        
        
        
        
    
    