from numpy import array,meshgrid,arange

from matplotlib import pyplot as p

#min c.T*x 
#Ax <= b

A =  array([[0,-1], [-1,-1], [-1,2], [1,-1]])
B = array([[-5], [-9],[0], [-3]])
c = array([[-1],[-2]])

targets = [-12,-14,-16]

#calculation for contours of c
def calc(target,x):
    return (target + x)/-2

X = arange(-11,11,.1)

for target in targets: 
    Y = calc(target,X)
    p.plot(X,Y,label="c=%d"%target)
    
p.xlabel("x_0")
p.ylabel("x_1")    

"""
b : blue
g : green
r : red
c : cyan
m : magenta
y : yellow
k : black
w : white
"""
colors = ['c','m','y','k']

#constraint lines
def calc(a,b,x):
    return (b[0]-a[0]*x)/a[1]
    
for color,a,b in zip(colors,A,B): 
    Y = calc(a,b,X)
    if a[1] < 0: 
        label ="%d$x_1$-%d$x_2$ <= %d"%(a[0],-a[1],b[0]) 
    else: 
        label ="%d$x_1$+%d$x_2$ <= %d"%(a[0],-a[1],b[0]) 
    p.plot(X,Y,label=label,c=color)
    p.fill_between(X,Y,0*Y,color = color,alpha=.3)
    
p.scatter([3,],[6,],c='k',label="optimum",s=40)    
    
p.axis([-10,10,4,10])     
p.legend(loc=3)
p.show()

