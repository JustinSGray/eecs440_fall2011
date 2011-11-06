from random import random

from numpy.random import binomial

p_c = {True:.5,False:.5}

p_r_G_c = {True:.8,False:.2}

p_s_G_c = {True:.2,False:.5}

p_w_G_s_r ={True:{True:.9,False:.8},False:{True:.6,False:.1}}

"""
p_c = {True:.5,False:.5}

p_r_G_c = {True:.8,False:.2}

p_s_G_c = {True:.1,False:.5}

p_w_G_s_r ={True:{True:.99,False:.90},False:{True:.90,False:0}}
"""

def coin_flip(threshold): 
    return binomial(1,threshold)
    
def sample(): 
    #p_c*p_r_G_c*p_s_G_c*p_w_G_s_r
    
    c = coin_flip(.5)
    s = coin_flip(p_s_G_c[c])
    r = coin_flip(p_r_G_c[c])
    w = coin_flip(p_w_G_s_r[s][r])
    
    return (c,s,r,w)
    
if __name__=="__main__": 
    from sys import argv
    
    iters = int(argv[1])
    
    i = 0
    j = 0
    while i <= iters: 
        c,s,r,w = sample()
        if (not c) and w: 
            i += 1 #total valid samples
            
            j += r #number of times it rained
        
    print "Pr(Rain=True|Cloudy=False,WetGrass=True): ", j/float(i)        
        
        
            
