from math import exp, floor, ceil
import random

from os.path import abspath

from shell.mldata import parse_c45,ExampleSet

DATA_PATH = "./prog1data"


def load_project_data(project_name,n_folds=5): 
    """returns two sets of data from the specified project. The first one
    contains 4/5 of the data for training. The second contains the remaining 1/5 of 
    the data for testing.""" 
    root_dir = abspath(DATA_PATH)
    
    data = parse_c45(project_name,root_dir)
    
    n_data = len(data)
    pos_data = []
    neg_data = []
    for ex in data:
        if ex[-1]: 
            pos_data.append(ex)
        else: 
            neg_data.append(ex)
            
    n_pos = len(pos_data)
    n_neg = len(neg_data)
            
    random.shuffle(pos_data)
    random.shuffle(neg_data)
    
    n_pos_fold = int(ceil(n_pos/float(n_folds))) #n fold cross validation
    n_neg_fold = int(ceil(n_neg/float(n_folds)))
    
    folds = []
    for i in range(0,n_folds): #n_folds folds
        pos_fold = pos_data[n_pos_fold*i:n_pos_fold*i+n_pos_fold]
        neg_fold = neg_data[n_neg_fold*i:n_neg_fold*i+n_neg_fold]
        
        #not sure you need this, but it seems like a bad idea to train a
        # all on positive then negative examples. It' can't hurt really. 
        # should not do anything for deterministic backprop, but will for 
        # stochastic backprop
        pos_fold.extend(neg_fold)
        random.shuffle(pos_fold)      
        folds.append(pos_fold)
        
    #create the different training and test set pairs
    fold_sets = []
    for i in range(0,n_folds): 
        test = folds.pop(i)
        train = []
        for fold in folds: 
            train.extend(fold)
        fold_sets.append((ExampleSet(train),ExampleSet(test)))
        folds.insert(i,test)
    return fold_sets
    

def act(x):
    """some activation function, given a value for x""" 
    if x < -400: return 0.0 #hack for numerical issues 
    return 1./(1.+exp(-x))
    
def d_act(y):
    """derivative of the act_function as a function of the value of act(x)""" 
    return  y*(1-y)  
    
    
class Node(object): 
    def __init__(self,n_connections,inp_node=False,act_func=act,d_act_func = d_act): 
        self.w = [random.uniform(-.1,.1) for x in range(0,n_connections)]
        self.dL_dw = [0 for x in range(0,n_connections)]
        #print "check: ", self.w
        self.x = 0
        self.n = 0
        #self.dL_dn = 0
        self.dh_dn = 0
        self.inp_node = inp_node
        self.act_func = act_func
        self.d_act_func = d_act_func 
        
    def h(self,n): 
        #print "check:", n
        self.n = n #store the n, for use with backprop
        if self.inp_node: 
            self.x = n
        else: 
            self.x = self.act_func(n)
        self.dh_dn = self.d_act_func(self.x)
        return self.x    
            
       
    
class ANN(object): 
    def __init__(self,training_data,n_hidden): 
        """creates a NN instance with n_inputs input nodes, n_hidden hidden nodes
        and one output node"""
        
        self.training_data = training_data
        
        #nested dict, first key is name of feature
        #second key is feature value, or "n_values"
        #data is the index of the place to put a 1 in the list 
        self.nominal_mapping = {} 
        #dictionary where key is feature name, and value is a two tuple of (mu,sigma)
        self.cont_mapping = {}
        
        n_data  = float(len(training_data))
        n_inputs = float(len(training_data.schema[1:-1]))
        
        for j,s in enumerate(training_data.schema[1:-1]): 
            #print s.type, s.name, s.values
            if s.type != 'CONTINUOUS' : 
                if s.type == "BINARY": 
                    s.values = [0,1]
                data = [i for i in range(0,len(s.values))]
                mu = sum(data)/n_data
                sigma = sum([(d-mu)**2/n_data for d in data])**.5
                
                self.nominal_mapping[s.name] = dict([(v,(i-mu)/sigma) for i,v in enumerate(s.values)])
            else: 
                data = [ex[j+1] for ex in training_data]
                mu = sum(data)/n_data
                sigma = sum([(d-mu)**2/n_data for d in data])**.5
                self.cont_mapping[s.name] = {'mu':mu,'sigma':sigma}
        
        #recalc length of inputs with new mapping
        n_inputs = len(self._input_vector(training_data[0]))
        
        self.nodes = []
        #initialize weights to small random values
        if n_hidden: 
            #input nodes
            self.nodes.append([Node(n_hidden,inp_node=True) for i in range(0,n_inputs)])  
                
            #hidden layer 
            self.nodes.append([Node(1) for i in range(0,n_hidden)])
            
            #output layer
            self.nodes.append([Node(0),])
        else: 
            #input nodes
            self.nodes.append([Node(1,inp_node=True) for i in range(0,n_inputs)])      
            #output layer
            self.nodes.append([Node(0),])
                
    def _input_vector(self,ex): 
        """takes an example, and returns an input array with all the nominal features
        expanded into k dimentional vectors with a 1 in the position equal to the 
        appropriate value, and all the continuous values normalized""" 
        vec = []
        for s,f in zip(ex.schema[1:-1],ex[1:-1]): 
            if s.type!='CONTINUOUS':
               map = self.nominal_mapping[s.name]
               vec.append(map[f])
            else:   
               map = self.cont_mapping[s.name]
               vec.append((f-map['mu'])/map['sigma'])
            
        return vec
            
    def train(self,max_iterations=0,gamma=.001,eta=.1): 
        """train the NN instance for at most max_iterations, with a learning rate 
        of eta, and a weight decay of gamma""" 
        iteration = 0
        while True: 
            for i,ex in enumerate(self.training_data):
                self.predict(ex)
                self._backprop(ex[-1],gamma,eta)
            iteration += 1 
            
            dL_dn = sum([node.dL_dw[0]*node.w[0]/node.x for node in self.nodes[-2]])
            #if iteration % 10 == 0: print "iteration :", iteration, dL_dn
            if (iteration >= 30000) or (max_iterations and iteration == max_iterations): 
                break
            if abs(dL_dn) < 1e-3: #this is convergence
                break
            #if abs(self.nodes[-1][0].dL_dn-prev_dL_dn) < 1e-15: #this might be convergence
            #    break 
            
            
        return iteration        
        
    def _backprop(self,output,gamma,eta): 
        """takes in a the value of the output expected at the output node"""
        #weights into output layer
        out_node = self.nodes[-1][0]
        out_node.dL_dn = (out_node.x-output)*out_node.dh_dn     
        for node in self.nodes[-2]: 
            for j,w in enumerate(node.w): 
                node.dL_dw[j] = out_node.dL_dn*node.x + gamma*w 
                node.w[j] -= node.dL_dw[j]          
        
        if len(self.nodes) > 2: #then there was a hidden layer, so update the weights into that one
            for j,node in enumerate(self.nodes[-2]): 
                for prev_node in self.nodes[-3]: 
                    prev_node.dL_dw[j] = node.dh_dn*prev_node.x*sum([dL_dw*w for dL_dw,w in zip(node.dL_dw,node.w)])
                    prev_node.w[j] -= prev_node.dL_dw[j]
        
          
         
                    
              
    def predict(self,example): 
       """gives the predicted value for the example activated across the NN instance""" 
       inputs = self._input_vector(example)
              
       #input layer
       for inp,node in zip(inputs,self.nodes[0]): 
           node.h(inp)
       #hidden layer
       for i,layer in enumerate(self.nodes[1:]):
           for j,node in enumerate(layer): 
               #print "test ",len([n.x*n.w[j] for n in self.nodes[i]]), 
               #print [("%0.3f"%n.w[j],"%0.6f"%(n.x)) for n in self.nodes[i]]
               #print
               node.h(sum([n.x*n.w[j] for n in self.nodes[i]]))
       return self.nodes[-1][0].x
                   
    
    
if __name__=="__main__": 
    import sys
    data_name = sys.argv[1]
    hidden_units = int(sys.argv[2])
    gamma = float(sys.argv[3])
    max_iterations = int(sys.argv[4])
    
    random.seed(12345)
    
    folds = load_project_data(data_name,5)
    networks = []
    
    TP = []
    TN = []
    FP = []
    FN = []
    
    accuracy = []
    precision = []
    recall = []
    results = []
    ROC_set = []
    
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
        
    
    for train_set,test_set in folds: 
        ann = ANN(train_set,hidden_units)
        n = ann.train(max_iterations,gamma=gamma)
        #print "trained in %d iterations"%n
        TP.append(0)
        TN.append(0)
        FP.append(0)
        FN.append(0)
        
        result = []
        for ex in test_set: 
            a = ann.predict(ex)
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
        
    def stats(results): 
        avg = sum(results)/float(len(results))
        sigma = sum([((r-avg)**2)/float(len(results)) for r in results])**.5 
        return avg,sigma
    
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
    from matplotlib import pyplot as p
    
    p.plot(FP_rate,TP_rate)
    p.show()    
        
          
        