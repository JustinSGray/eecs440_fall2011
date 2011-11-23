from load import load_project_data

from math import pi
import random

from numpy import array, dot, exp, zeros, append, log, mean, std
from numpy.linalg import norm
from numpy.random import rand, seed
from scipy.optimize import fmin_powell as fmin

    
class Logistic(object):

    def __init__(self,examples,gamma): 
        self.lamb = lamb
        self.train(examples)

    def _likelihood(self,example,W): 
        y = example[-1]
        X = append((example[1:-1]-self.mean)/self.std,[1]) #added on one, so you can just use the w array
        if y:
            tmp = log(1/(1+exp(-dot(W,X))))
        else:          
            tmp = log(1-(1/(1+exp(-dot(W,X)))))
        return tmp
            
            
        
        #return y*dot(W,X) - log(1+exp(dot(W,X))) 
                
    def _f(self,W): 
        l = 0 
        for ex in self.training_examples: 
            l+= self._likelihood(ex,W)
            
        #print -l + self.lamb*.5*norm(W,2)**2, W
        return -l + self.lamb*.5*norm(W,2)**2           
            
    def train(self,examples): 
        self.training_examples = array(examples.to_float())
        self.mean = mean(self.training_examples[:,1:-1],0)
        self.std = std(self.training_examples[:,1:-1],0)

        self.W = rand(self.training_examples.shape[1]-1)*.001 #small intial values
        #optimization        
        self.W = array(fmin(self._f,self.W,disp=False))
        
    def predict(self,example):
        ex = example.to_float()
        X = append((ex[1:-1]-self.mean)/self.std,[1])
        
        return 1/(1+exp(-dot(self.W,X)))

if __name__=="__main__": 
    import sys
    data_name = sys.argv[1]
    lamb = float(sys.argv[2])
    
    import time
    
    seed(12345)
    random.seed(12345)
    
    def stats(results): 
        avg = sum(results)/float(len(results))
        sigma = sum([((r-avg)**2)/float(len(results)) for r in results])**.5 
        return avg,sigma
        
    def cont_table(test_set,results,thresh):
         tp = 0
         fn = 0
         fp = 0
         tn = 0
         for e,r in zip(test_set,results): 
            a = r > thresh
            if e[-1] and a: tp+=1
            elif e[-1]: fn+=1
            elif a: fp+=1
            else: tn+=1
         return tp,fn,fp,tn   
        
    
    folds = load_project_data(data_name,5)
    
    TP = []
    TN = []
    FP = []
    FN = []
    
    accuracy = []
    precision = []
    recall = []
    results = []
    ROC_set = []
    
    
    for train_set,test_set in folds: 
        lr = Logistic(train_set,lamb)
        #print "trained in %d iterations"%n
        TP.append(0)
        TN.append(0)
        FP.append(0)
        FN.append(0)
        
        result = []
        for ex in test_set: 
            a = lr.predict(ex)
            result.append(a)
            
        tp,fn,fp,tn = cont_table(test_set,result,.5)   
        TP.append(tp)
        FN.append(fn)
        FP.append(fp)
        TN.append(tn)
        
        ROC_set.extend(test_set)
        results.extend(result)  
          
        accuracy.append((TP[-1]+TN[-1])/float((TP[-1]+TN[-1]+FP[-1]+FN[-1]))) 
        if TP[-1]:     
            precision.append(TP[-1]/float(TP[-1]+FP[-1]))
            recall.append(TP[-1]/float(TP[-1]+FN[-1]))
        else: 
            precision.append(0.0)
            recall.append(0.0)
            
            
    print "t-test data: " 
    from numpy import array
    print 1-array(accuracy)   
    
    mu,sigma = stats(accuracy)
    print "Accuracy: %0.3f, %0.3f"%(mu,sigma) 
    print
    mu,sigma = stats(precision)
    print "Precision: %0.3f, %0.3f"%(mu,sigma) 
    print
    mu,sigma = stats(recall)
    print "Recall: %0.3f, %0.3f"%(mu,sigma)    
    
    
    TP_rate = [0.0]
    FP_rate = [0.0]
    #need to sort the data by result
    results,ROC_set = zip(*sorted(zip(results,ROC_set),reverse=True))
    #calculation for AROC
    for r in results: 
        tp,fn,fp,tn = cont_table(ROC_set,results,r)
        
        if fp:   
            FP_rate.append(fp/float(fp+tn))
        else: FP_rate.append(0.0)
        if tp: 
            TP_rate.append(tp/float(tp+fn))
        else: TP_rate.append(0.0) 
    #get the last one
    tp,fn,fp,tn = cont_table(ROC_set,results,results[-1]*.9)
        
    if fp:   
        FP_rate.append(fp/float(fp+tn))
    else: FP_rate.append(0.0)
    if tp: 
        TP_rate.append(tp/float(tp+fn))
    else: TP_rate.append(0.0) 
    
        
    aroc = 0  
    for p1,p2 in zip(zip(FP_rate[0:-1],TP_rate[0:-1]),zip(FP_rate[1:],TP_rate[1:])):
        #print p2[0],p1[0]
        aroc += (p2[0]-p1[0])*(p2[1]+p1[1])/2.0  
    print 
    print "AROC: %0.3f"%aroc,     
    #from matplotlib import pyplot as p
    
    #p.plot(FP_rate,TP_rate)
    #p.show()    
        
    
