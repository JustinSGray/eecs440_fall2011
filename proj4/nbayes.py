from load import load_project_data

from math import pi, exp


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
        

if __name__=="__main__": 

    import sys
    data_name = sys.argv[1]
    C = float(sys.argv[2])
    
    import time
    
    random.seed(12345)
    
    folds = load_project_data(data_name,5)
