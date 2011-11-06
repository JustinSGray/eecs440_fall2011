from load import load_project_data

from math import pi
import random

from numpy import array, dot, exp, zeros, append, log
from numpy.linalg import norm
from numpy.random import rand, seed
from scipy.optimize import fmin_cg

    
class Logistic(object):

    def _likelihood(self,example,W): 
        y = example[-1]
        X = append(example[1:-1],[1]) #added on one, so you can just use the w array
        
        if y: 
            return log(1/(1+exp(dot(W,X))))
        else:     
            return log(1-(1/(1+exp(dot(W,X)))))
        
        #return y*dot(W,X) - log(1+exp(dot(W,X))) 
                
    def _f(self,W): 
        l = 0 
        for ex in self.training_examples: 
            l+= self._likelihood(ex,W)
        
        return -l + .5*norm(W,2)**2           
            
    def train(self,examples): 
        
        self.training_examples = array(examples.to_float())
        self.W = rand(self.training_examples.shape[1]-1)*.001 #small intial values
                
        print fmin_cg(self._f,self.W)
        
    def predict(self,example):
        pass          

if __name__=="__main__": 
    import sys
    data_name = sys.argv[1]
    lamb = float(sys.argv[2])
    
    import time
    
    seed(12345)
    random.seed(12345)
    
    folds = load_project_data(data_name,3)
    
    lr = Logistic()
    
    lr.train(folds[0][0])
    
