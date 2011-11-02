from numpy import matrix 
from numpy.linalg import eig

A = matrix([[.4,.4,.2],[.4,.6, 0.],[.8,0,.2]])

D,V = eig(A.T)

print "Tristan: ", (A.T**30)

print 
print 
print "Eigen: ",V[:,0]*1/sum([x[0] for x in V[:,0]])
