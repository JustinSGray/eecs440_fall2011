from os.path import abspath
import random

from math import floor, ceil

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