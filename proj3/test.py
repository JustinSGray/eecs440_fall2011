from numpy import zeros,dot, matrix, vstack, hstack,identity, array, ones

n_features = 3
n_training = 2

n_x = 1+ n_features+ n_training
                        
p_1 = zeros((n_x,1))

p_21 = zeros((1,n_features))
p_22 = identity(n_features)
p_23 = zeros((n_training,n_features))
p_2  = vstack((p_21,p_22,p_23))

p_3  = zeros((n_x,n_training))

P = hstack((p_1,p_2,p_3))

print P.shape
print 
print P

test = array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
print test[:,1:-1]

a = array([1,2])

aa = vstack((a,)*3)
print -1*aa*test[:,1:-1]

print ones((3,1))
