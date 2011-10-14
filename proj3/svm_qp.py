import random

from numpy import zeros,dot, vstack, hstack,identity, ones, array,average,std

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
        
        #scaling data for unit variance
        self.mu = ones(n_features)
        self.sigma = ones(n_features)
        for i,column in enumerate(self.training_set[:,1:-1].T): 
            self.mu[i] = average(column)
            self.sigma[i] = std(column)
            self.training_set[:,i+1] = (self.training_set[:,i+1] - self.mu[i])/self.sigma[i]
            
        #print self.mu
        #print self.sigma
        #print self.training_set
       
        
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
        
        solvers.options['show_progress'] = False
        sol=solvers.qp(P, q, G, h)
        
        #print sol['x'].T
        
        self.b = sol['x'].T[0]
        self.w = array(sol['x'].T[1:n_features+1]).T
        self.p = array(sol['x'].T[n_features+1:]).T
        
        #print "test", self.b
        #print "test", self.w
        #print "test", self.p
        
        return (self.b,self.w,self.p)
        
        
    def predict(self,ex): 
        x= array(ex.to_float()[1:-1])
        for i,(mu,sigma) in enumerate(zip(self.mu,self.sigma)): 
            x[i] = (x[i]-mu)/sigma
        return (dot(self.w,x) + self.b)[0]
            
               
        
        
if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    C = float(sys.argv[2])
    
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
    
    for train_set,test_set in folds: 
        s = SVM(train_set,C)
        results = [s.predict(ex) for ex in test_set]  
        TP,FN,FP,TN = cont_table(test_set,results,0)
        
        accuracy.append((TP+TN)/float((TP+TN+FP+FN))) 
        if TP:     
            precision.append(TP/float(TP+FP))
            recall.append(TP/float(TP+FN))
        else: 
            precision.append(0.0)
            recall.append(0.0)
            
            
            
    mu,sigma = stats(accuracy)
    print "Accuracy: %0.3f, %0.3f"%(mu,sigma) 
    print
    mu,sigma = stats(precision)
    print "Precision: %0.3f, %0.3f"%(mu,sigma) 
    print
    mu,sigma = stats(recall)
    print "Recall: %0.3f, %0.3f"%(mu,sigma)         
        
        
        
         
           
        
        
        
        
        
        
    
    