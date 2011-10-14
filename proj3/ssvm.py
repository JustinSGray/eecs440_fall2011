import random

from numpy import zeros,dot, vstack, hstack,identity,\
                  ones, array,average,std, matrix, exp, log
from numpy.linalg import norm

from load import load_project_data
        
class SSVM(object): 
    
    def __init__(self,training_set,nu):
        self.nu = nu
        self.train(training_set)
        
    def _p(self,z,alpha): 
        return z + 1./alpha*log(1+exp(-alpha*z))
        
    def _Phi(self,w,gamma,alpha): 
        p = self._p(self.e-self.D*(self.A*w-gamma*self.e),alpha)
        return (self.nu*norm(p)**2 + (w.T*w)[0,0]+gamma**2)/2.0
        
    def _grad_Phi(self,w,gamma,alpha): 
        center = self._Phi(w,gamma,alpha)
        d_phi = []
        dd_phi = []          
        #partial w.r.t. w  
        for i,value in enumerate(w):
            right_w = array(w)
            if right_w[i,0]: #non, zero
                right_w[i,0] += .01*right_w[i,0]
            else: 
                right_w[i,0] = .01    
            
            right = self._Phi(right_w,gamma,alpha)
            
            left_w = array(w)
            if left_w[i,0]: #non, zero
                left_w[i,0] -= .01*left_w[i,0]   
            else: 
                left_w[i,0] = -.01
                    
            left = self._Phi(left_w,gamma,alpha)
            
            d_phi.append((right-left)/(right_w[i,0]-left_w[i,0]))
            dd_phi.append((right-2*center+left)/(right_w[i,0]-left_w[i,0])**2)

        #partial w.r.t. gamma 
        right_gamma = gamma + gamma*.01
        right = self._Phi(w,right_gamma,alpha)
        
        left_gamma = gamma - gamma*.01   
        left = self._Phi(w,left_gamma,alpha)
        
        d_phi.append((right-left)/(right_gamma-left_gamma))
        dd_phi.append((right-2*center+left)/(right_gamma-left_gamma)**2)
        
        return center,array(d_phi),array(dd_phi)
                    
                         
    def train(self,training_set):
    
        self.training_set = array(training_set.to_float()) 
        
        
        n = len(self.training_set[0][1:-1])
        m = len(self.training_set)
        
        #scaling data for unit variance
        self.mu = ones(n)
        self.sigma = ones(n)
        for i,column in enumerate(self.training_set[:,1:-1].T): 
            self.mu[i] = average(column)
            self.sigma[i] = std(column)
            self.training_set[:,i+1] = (self.training_set[:,i+1] - self.mu[i])/self.sigma[i]
            
        self.A = matrix(self.training_set[:,1:-1])
        
        #self.e = matrix(self.training_set[:,-1]).T
        #self.e = (self.e==0).choose(self.e,-1) #replace all 0's with -1
        self.e = matrix(ones((m,1)))

        self.D = matrix(array(identity(m))*array(hstack((self.e,)*m)))
        
        self.w = matrix(zeros((n,1)))
        
        self.gamma = 1
        
        #tmp = self._p(e-D*(A*w-gamma*e),5)
        center, d_Phi, dd_Phi =  self._grad_Phi(self.w,self.gamma,2) 
        #print -d_Phi/dd_Phi
        while any(abs(d_P) > .01 for d_P in d_Phi): 
            direction =  array([-d_Phi/dd_Phi]).T
            self.w = self.w+.5*direction[0:n] 
            self.gamma = self.gamma + .25*direction[n,0]
            
            center, d_Phi, dd_Phi =  self._grad_Phi(self.w,self.gamma,2) 
            
            
        exit()
    def predict(self,ex): 
        x= array(ex.to_float()[1:-1])
        for i,(mu,sigma) in enumerate(zip(self.mu,self.sigma)): 
            x[i] = (x[i]-mu)/sigma
        
        return (dot(self.w,x) + self.b)[0]
            
               
        
        
if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    nu = float(sys.argv[2])
    
    random.seed(12345)
    
    folds = load_project_data(data_name,3)
        
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
        s = SSVM(train_set,nu)
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
        
        
        
         
           
        
        
        
        
        
        
    
    