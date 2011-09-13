from numpy import random as rand, array,zeros

def y(x): 
    return 1-x

TRIALS = 1000000
SD = zeros(TRIALS)
for i in range(0,TRIALS):
    X = rand.uniform(0,1,2)
    Y = y(X)

    sd = (X[1]-X[0])**2 + (Y[1]-Y[0])**2
    SD[i] = sd

print SD.mean()


