from load import load_project_data

from math import pi, exp, log

import random

from numpy import array


class DiscreteTrainer(object): 
    
    def train(self,X,Y,allowed_values,m=0): 
        
        
        self.pbins = dict([(v,0) for v in allowed_values])
        n_y1 = 0
        
        self.nbins = dict(self.pbins)
        n_y0 = 0
        
        for x,y in zip(X,Y): 
            if y: 
                self.pbins[x] += 1
                n_y1 += 1
            else: 
                self.nbins[x] += 1  
                n_y0 += 1
                
        n_y1 = float(n_y1)
        n_y0 = float(n_y0) 
        
        if m < 0: 
            m = float(len(allowed_values))
        p = 1/float(len(allowed_values)) #prior probibility for smoothing
             
        for value in allowed_values:           
            self.pbins[value] = (self.pbins[value]+m*p)/(n_y1+m)
            self.nbins[value] = (self.nbins[value] +m*p)/(n_y0+m)
        
    def cond_prob(self,x): 
        """returns a 2-tuple of (p(x|y=1),p(x|y=0))"""
        return (self.pbins[x],self.nbins[x])
       
       
       
class ContinousTrainer(object): 
    def train(self,X,Y): 
        self.pbins = []
        n_y1 = 0
        
        self.nbins = []
        n_y0 = 0
        
        for x,y in zip(X,Y): 
            if y: 
                self.pbins.append(x)
                n_y1 += 1
            else: 
                self.nbins.append(x)
                n_y0 += 1 
                
        n_y1 = float(n_y1)
        self.pmu = sum(self.pbins)/n_y1
        self.pvar = sum([(x-self.pmu)**2 for x in self.pbins])/n_y1 #sigma**2
        if self.pvar < .01: self.pvar = .01 #minimum variance
        
        n_y0 = float(n_y0) 
        self.nmu = sum(self.nbins)/n_y0
        self.nvar = sum([(x-self.nmu)**2 for x in self.nbins])/n_y0 #sigma**2  
        if self.nvar < .01: self.nvar = .01 #minimum variance
    
    def cond_prob(self,x):
        """returns a 2-tuple of (p(x|y=1),p(x|y=0))"""
        pp = 1/(2*pi*self.pvar)**.5*exp(-.5*(x-self.pmu)**2/self.pvar)
        np = 1/(2*pi*self.nvar)**.5*exp(-.5*(x-self.nmu)**2/self.nvar)
        
        return (pp,np)
        
class NaiveBayes(object): 

    def __init__(self,m,training_data):
        if training_data: 
            self.train(m,training_data)  
            
    def train(self,m,training_data): 
        
        Y = [x[-1] for x in training_data]
        n_y1 = sum(Y)
        n_y = float(len(Y))
        
        self.py1 = n_y1/n_y
        self.py0 = (n_y-n_y1)/n_y
        self.nodes = []
        for i,s in enumerate(training_data.schema[1:-1]): 
            X = [x[i+1] for x in training_data]
            if s.type=="CONTINUOUS": 
                node = ContinousTrainer()
                node.train(X,Y)
            else: 
                if s.type=="BINARY": 
                    s.values = (0,1)
                    
                node = DiscreteTrainer()
                node.train(X,Y,s.values,m)
                
            self.nodes.append(node)    
                
        
    def predict(self,X): 
        X = X[1:-1]
        pp = log(self.py1)
        np = log(self.py0)
        
        for i,x in enumerate(X): 
            cpp,cpn = self.nodes[i].cond_prob(x)
            #print cpp, cpn
            try: 
                pp += log(cpp)
            except ValueError: 
                pass #if there is no probability, just skip it 
            try:     
                np += log(cpn)
            except ValueError: 
                pass #same as above
        
        
        return (pp>np,pp)   

if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    m = float(sys.argv[2])
    
    import time
    
    random.seed(12345)
    
    folds = load_project_data(data_name,5)
        
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
        
    def stats(results): 
        avg = sum(results)/float(len(results))
        sigma = sum([((r-avg)**2)/float(len(results)) for r in results])**.5 
        return avg,sigma
        
        
    accuracy = []
    precision = []
    recall = []    
    
    results = []
    ROC_set = []
    for train_set,test_set in folds: 
        nb = NaiveBayes(m,train_set)
        result = [nb.predict(ex) for ex in test_set]  
        results.extend(result)
        ROC_set.extend(test_set)
        TP,FN,FP,TN = cont_table(test_set,[x[0] for x in result],0)
        
        accuracy.append((TP+TN)/float((TP+TN+FP+FN))) 
        if TP:     
            precision.append(TP/float(TP+FP))
            recall.append(TP/float(TP+FN))
        else: 
            precision.append(0.0)
            recall.append(0.0)
            
            
    print "t-test data: " 
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
    
    
    results = [r[1] for r in results]
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

