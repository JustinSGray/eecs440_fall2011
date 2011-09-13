import sys
from os.path import abspath,join
import random 
from math import floor, ceil, log

from shell.mldata import parse_c45,ExampleSet

DATA_PATH = "./prog1data"

MAX_DEPTH = 0 #default to 0

def load_project_data(project_name): 
    """returns two sets of data from the specified project. The first one
    contains 4/5 of the data for training. The second contains the remaining 1/5 of 
    the data for testing.""" 
    root_dir = abspath(DATA_PATH)
    
    data = parse_c45(project_name,root_dir)
    
    n_data = len(data)
    n_train,n_test = int(floor(4/5.0*n_data)),int(ceil(1/5.0*n_data))
    
    train_choices = set(random.sample(xrange(n_data),n_train))
        
    train_data, test_data = [],[]
    for i,ex in enumerate(data): 
        if i in train_choices: 
            train_data.append(ex)
        else: 
            test_data.append(ex)
    
    return ExampleSet(train_data),ExampleSet(test_data)
    
    
class Binner(object): #default implementation
    def __call__(self,value): 
        return value   

class Node(object): 
    
    def __init__(self): 
        self.binner = Binner()
        self.children = {}
        self.attr_index = 0 #each node classifies according to a specific attr index
        
        self.is_leaf = False #determines if you're a leaf node or not
        self.classifier = None 
        
    def partition_data(self,ex_set,attr_index): 
        """returns a 3-tuple of (H_x,H_y_x,partitioned_data)"""  
        part_data = {}
        n_data = float(len(ex_set))
        
        for ex in ex_set: 
               #converts the value to a binned value, really only needed for continuous attrs though
               bin = self.binner(ex[attr_index])    
               part_data.setdefault(bin,[]).append(ex)
               
        H_y_x = 0
        H_x = 0 
        for bin,data in part_data.iteritems(): 
            n_bin = float(len(data))
            
            p_bin = n_bin/n_data
            p_plus = sum([1 for ex in data if ex[-1]])/n_bin
            p_minus = 1.0 - p_plus
            
            try: 
                H_y_x_bin = -p_plus*log(p_plus,2)-p_minus*log(p_minus,2)
            except ValueError: #0log0 defined as 0 
                H_y_x_bin = 0  
                
            H_x += -1*p_bin*log(p_bin,2) 
                
            H_y_x += p_bin*H_y_x_bin
        
        return H_x,H_y_x,part_data  
              
              
    def max_GR(self,ex_set,attr_set): 
        """returns a 2-tuple of (attr_index,part_data) for the attr with the max GR"""
        n_data = float(len(ex_set))
        p_plus = sum([1 for ex in ex_set if ex[-1]])/n_data
        p_minus = 1.0 - p_plus
        
        try: 
            H_y = -p_plus*log(p_plus,2)-p_minus*log(p_minus,2) 
        except ValueError: 
            H_y = 0       
        
        GR = 0
        max_GR = (None,None)
        for attr_index in attr_set: 
            H_x, H_y_x, part_data = self.partition_data(ex_set,attr_index)
            gain_ratio = (H_y - H_y_x)/H_x
            if gain_ratio > GR: 
                GR = gain_ratio
                max_GR = attr_index,part_data
                 
        return max_GR
        
        
    def check_ex_set(self,ex_set,attr_set):     
        """checks data to make sure it's partitionable. returns a 2-tuple
        of (most common classifier,partitionalbe)"""
        #sub_data is not partitionable if: 
        #  1) Data is homogeneous in classifier
        #  2) Data is homogeneous in attr_set 
        #  3) There are no attrs left to partition on
        
        n_data = len(ex_set)
        n_half_data = n_data/2
        n_pos = 0
        
        #both checks are True if homogeneous data
        attr_check = True 
        classifier_check = True
        for i,ex in enumerate(ex_set): 
            n_pos += ex[-1]
            
            if attr_check: #still looks homogeneous for attrs
                attr_check = all([ex[attr]==ex_set[0][attr] for attr in attr_set])
            if classifier_check: #still looks homogeneous for classifiers    
                classifier_check = ex[-1] == ex_set[0][-1]
            
            #note: you can stop as soon as both check trips to false, 
            #   but you might need mcc if you reached max depth, so keep going
            #   at least untill halfway through the data, so you can calc mcc 
            if ((not attr_check) and (not classifier_check)) and i > n_half_data: 
                break
        #calc mcc
        if n_pos == n_half_data: #pick randomly
            mcc = bool(random.randint(0,1))
        else: 
            mcc = n_pos > n_half_data    
            
        
        return mcc, ((not attr_check) and (not classifier_check))
        
    def train(self,ex_set,attr_set,depth=0): 
        """trains a tree, based on the given data. depth is used to track tree depth 
        so that stopping conditions can be enforced"""   
        
        mcc,partable = self.check_ex_set(ex_set,attr_set)
        
        if partable and not depth == MAX_DEPTH: 
            attr,part_data = max_GR(ex_set,attr_set)
            self.attr_index = attr
            new_attr_set = attr_set[:]
            new_attr_set.remove(attr) 
        
            for feature,sub_data in part_data.iteritems():
                self.children[feature] = Node()
                self.children[feature].train(sub_data,new_attr_set,depth+1)
                
        else: 
            self.is_leaf = True
            self.classifier = mcc #most common classifier
            
            
            
    def predict(self,example): 
            
        if self.is_leaf: 
            return self.classifier
        return self.children[example[self.attr_index]]
            
    def shape(self): 
        """returns a 2-tuple of (size,depth)"""
        size = 1 
        depth = 0
        
        
        if self.children: 
            c_depths = []
            for feature,child in self.children.iteritems(): 
                c_shape = child.shape()
                size += c_shape[0]
                c_depths.append(c_shape[1])
             
            depth += 1+max(c_depths)
                   
        return size,depth        
    
    
if __name__=="__main__": 
    random.seed(12345)
    
    problem_name = sys.argv[1]
    MAX_DEPTH = int(sys.argv[2]) #non negative int, if 0 then grow to full depth
    
    print "problem_name: ", problem_name
    print "************************************************************"
    print 
    
    train_data,test_data = load_project_data(problem_name)

    print len(train_data) + len(test_data)    