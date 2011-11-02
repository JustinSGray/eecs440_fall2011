from random import random

p_c = {True:.5,False:.5}

p_r_G_c = {True:.8,False:.2}

p_s_G_c = {True:.2,False:.5}

p_w_G_s_r ={True:{True:.9,False:.8},False:{True:.6,False:.1}}

def coin_flip(threshold): 
    flip = random()
    return flip >= threshold
    
def sample(): 
    #p_c*p_r_G_c*p_s_G_c*p_w_G_s_r
    
    c = coin_flip(p_c[True])
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
            i += 1
            j += r
        
    #print i < j
    #print samples        
    print "Pr(Rain=True|Cloudy=false,WetGrass=True): ", j/float(i)        
        
        
            
