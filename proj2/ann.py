from math import e, floor, ceil
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
    return 1./(1.+e**-x)
    
def d_act(y):
    """derivative of the act_function as a function of the value of act(x)""" 
    return  y*(1-y)  
    
    
class Node(object): 
    def __init__(self,n_connections,inp_node=False,act_func=act,d_act_func = d_act): 
        self.w = [random.uniform(-.1,.1) for x in range(0,n_connections)]
        #print "check: ", self.w
        self.x = 0
        self.n = 0
        self.dL_dn = 0
        self.dh_dn = 0
        self.inp_node = inp_node
        self.act_func = act_func
        self.d_act_func = d_act_func 
        
    def h(self,n): 
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
        
        n_data = n_data = float(len(training_data))
        n_inputs = float(len(training_data.schema[1:-1]))
        
        for j,s in enumerate(training_data.schema[1:-1]): 
            #print s.type, s.name, s.values
            if s.type != 'CONTINUOUS' : 
                data = [i for i in range(0,len(s.values))]
                mu = sum(data)/n_data
                sig_data = [(d-mu)**2 for d in data]
                sigma = (sum(sig_data)/n_data)**.5
                
                self.nominal_mapping[s.name] = dict([(v,(i-mu)/sigma) for i,v in enumerate(s.values)])
            else: 
                data = [ex[j+1] for ex in training_data]
                mu = sum(data)/n_data
                sig_data = [(d-mu)**2 for d in data]
                sigma = (sum(sig_data)/n_data)**.5
                self.cont_mapping[s.name] = {'mu':mu,'sigma':sigma}
        
        #recalc length of inputs with new mapping
        n_inputs = len(self._input_vector(training_data[0]))
        
        self.nodes = []
        #initialize weights to small random values
        #input nodes
        self.nodes.append([Node(n_hidden,inp_node=True) for i in range(0,n_inputs)])  
            
        #hidden layer 
        self.nodes.append([Node(1) for i in range(0,n_hidden)])
        
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
       for iter in xrange(0,max_iterations):  
           for ex in self.training_data: 
               self.predict(ex)
               self._backprop(ex[-1],gamma,eta)
   
    def _backprop(self,output,gamma,eta): 
        """takes in a the value of the output expected at the output node"""
        #output layer
        out_node = self.nodes[2][0]
        out_node.dL_dn = (out_node.x-output)*out_node.dh_dn
        
        #back layers
        layers = self.nodes[0:-1]
        layers.reverse()
        n_layers = len(layers)
        for i,layer in enumerate(layers):
            for node in self.nodes[1]: 
                dL_dn =[]
                for j,w in enumerate(node.w): 
                    dL_dn.append(self.nodes[n_layers-1][j].dL_dn*w*node.x)
                    dL_dw = out_node.dL_dn*node.x
                    node.w[j] -= (eta*dL_dw + eta*gamma*w)
                node.dL_dn = sum(dL_dn)
            
                    
              
    def predict(self,example): 
       """gives the predicted value for the example activated across the NN instance""" 
       inputs = self._input_vector(example)
              
       #input layer
       for inp,node in zip(inputs,self.nodes[0]): 
           node.h(inp)
           #print "(",inp,node.h(inp),") ",
           
       #hidden layer
       for i,layer in enumerate(self.nodes[1:]):
           #print "\n  ",
           for j,node in enumerate(layer): 
               #print "(",[n.w[j] for n in self.nodes[i]],") ",
               node.h(sum([n.x*n.w[j] for n in self.nodes[i]]))
           #print
       return self.nodes[-1][0].x > .5
                   
    
    
if __name__=="__main__": 
    import sys
    data_name = sys.argv[1]
    hidden_units = int(sys.argv[2])
    gamma = float(sys.argv[3])
    max_iterations = int(sys.argv[4])
    
    random.seed(12345)
    
    folds = load_project_data(data_name,3)
    networks = []
    for train_set,test_set in folds: 
        ann = ANN(train_set,hidden_units)
        networks.append(ann)
        ann.train(max_iterations,gamma=gamma)

    
    #for ex in folds[0]: 
    #    print ann.predict(ex),ex[-1]
    
     
        