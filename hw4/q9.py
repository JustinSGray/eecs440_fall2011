from math import e
from numpy import meshgrid, arange, array

def sig(u): 
    return 1./(1.+e**(-u))
       
    
inputs = [0,0]
hidden = [0,0]
output = [0,]        

weights = [[[1,1],[1,1]],
           [[1,1],]
          ]
           
outputs = [[0,0],[0,0]]

def activate(x): 
    for i,val in enumerate(x): 
        
        outputs[0][i] = sig(val) 
    
    mult = .01   
    for j in range(0,len(hidden)): 
        n = sum([mult*x*w for x,w in zip(outputs[0],weights[0][j])]) 
        outputs[1][j] = sig(n)
    
    n = sum([mult*x*w for x,w in zip(outputs[1],weights[1][0])])    
    return sig(n)
    
    
if __name__=="__main__": 
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    x = arange(-5,5,.05)
    
    X,Y = meshgrid(x,x)
    Z = X.copy()
       
    for i,a in enumerate(x): 
        for j,b in enumerate(x): 
            Z[i][j] = activate([a,b])
    
          
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0) 
            
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()        
                   