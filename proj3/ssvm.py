import random

from numpy import zeros,dot, vstack, hstack,identity,\
                  ones, array,average,std, matrix, exp, log, log2, \
                  seterr, sign, nonzero, transpose
                  
                  
from scipy.sparse import identity as spident, spdiags                
                  


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
        
    def _grad(self,w,gamma):
        C = (self.D*self.A)
        d = (self.D*self.e)
        
        temp = C*w- gamma*d # D(Aw0 -e \gamma0)
        rv =  self.e - temp # e - D(Aw0 - e \gamma0)
        #calculate the Hessian
        H = (self.e + sign(rv))/2;
        T = identity(H.shape[0])
        SH = C.T*T
        P = SH*C
        q = SH*d
        
        #Q is the hessian
        Q = identity(self.w.shape[0]+1) + self.nu*vstack(( hstack((P,-q)) , hstack((-q.T,matrix([norm(H)]))) ))
        
        #calculate the gradient
        prv = (rv<0).choose(rv,0) #plus function
        
        gradz = vstack(((w - self.nu*C.T*prv), gamma+self.nu*d.T*prv))
        return matrix(gradz).T, Q 
       
              
    def _armijo(self,w,gamma,alpha,direction,gap): 
        step = 1; n = len(w) 
        obj1 = self._Phi(w,gamma,alpha)
        w2 = w + step*direction[0:n] 
        gamma2 = gamma +step*direction[n,0];
        obj2 = self._Phi(w2,gamma2,alpha)
        diff = obj1 - obj2
        
        while diff < -.05*step*gap: 
            step = 0.5*step;
            w2 = w + step*direction[0:n] 
            gamma2 = gamma +step*direction[n,0];
            obj2 = self._Phi(w2,gamma2,alpha)
            diff = obj1 - obj2
            
        return step    
                
                         
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
        
        
        self.e = matrix(ones((m,1)))

        y = self.training_set[:,-1]
        y = array([(y==0).choose(y,-1)]).T
           
        self.D = matrix(array(identity(m))*array(hstack((y,)*m)))
        
        self.w = matrix(zeros((n,1)))
        
        self.gamma = 0
        
        alpha = 5
        
        #tmp = self._p(e-D*(A*w-gamma*e),5)
        #center, d_Phi, dd_Phi =  self._grad_Phi(self.w,self.gamma,alpha) 
        d_Phi, dd_Phi = self._grad(self.w,self.gamma)
        #print -d_Phi/dd_Phi
        direction = dd_Phi.I*-1*d_Phi.T
        gap = direction.T*d_Phi.T
        step = self._armijo(self.w,self.gamma,alpha,direction,gap)
        
        #print d_Phi
        while step*(d_Phi*d_Phi.T)[0,0] >= 0.01:

            self.w = self.w+step*direction[0:n] 
            self.gamma = self.gamma + step*direction[n,0]
            d_Phi, dd_Phi = self._grad(self.w,self.gamma)
            print step*(d_Phi*d_Phi.T)[0,0]     
            direction = dd_Phi.I*-1*d_Phi.T
            gap = direction.T*d_Phi.T
            step = self._armijo(self.w,self.gamma,alpha,direction,gap)
 
            
        print "test",  sum(self.w)/len(self.w), self.gamma
        #exit()
    def predict(self,ex): 
        x= array(ex.to_float()[1:-1])
        for i,(mu,sigma) in enumerate(zip(self.mu,self.sigma)): 
            x[i] = (x[i]-mu)/sigma
        
        return (dot(self.w.T,x) - self.gamma)[0,0]
            
               
        
        
if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    nu = float(sys.argv[2])
        
    seterr(all='ignore')
    
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
        
        #print results
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
        
        
        
         
           
        
        
        
        
        
        
    
    