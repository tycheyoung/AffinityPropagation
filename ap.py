from numpy import median
from mpmath import mpf
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

iter = 50
lbda = 0.9  # damping

with open("ToyProblemData.txt") as textFile:
    locationPoint = [line.split() for line in textFile]
N = len(locationPoint)
locationPoint = [[float(y) for y in x] for x in locationPoint]
S = [[0]*N for i in range(N)]
A = [[0]*N for i in range(N)]
R = [[0]*N for i in range(N)]

for i in range(0,N-1,1):
    for j in range(i+1,N,1):
        S[i][j] = -((locationPoint[i][0]-locationPoint[j][0])*(locationPoint[i][0]-locationPoint[j][0])+(locationPoint[i][1]-locationPoint[j][1])*(locationPoint[i][1]-locationPoint[j][1]))
        S[j][i] = S[i][j]

for i in range(0,N,1):
    S[i][i] = median(S)

for m in range(0,iter,1):
    # update responsibility
    for i in range(0,N,1):
        for k in range(0,N,1):
            maxi = mpf("-1e100")
            for kk in range(0,k,1):
                if(S[i][kk]+A[i][kk]>maxi):
                    maxi = S[i][kk]+A[i][kk]
            for kk in range(k+1,N,1):
                if(S[i][kk]+A[i][kk]>maxi): 
                    maxi = S[i][kk]+A[i][kk]
            R[i][k] = (1-lbda)*(S[i][k] - maxi) + lbda*R[i][k]
    # update availability
    for i in range(0,N,1):
        for k in range(0,N,1):
            if(i==k):
                sum = 0.0
                for ii in range(0,i,1):
                    sum += max(0.0, R[ii][k])
                for ii in range(i+1,N,1):
                    sum += max(0.0, R[ii][k])
                A[i][k] = (1-lbda)*sum + lbda*A[i][k]
            else:
                sum = 0.0
                maxik = max(i, k)
                minik = min(i, k)
                for ii in range(0,minik,1):
                    sum += max(0.0, R[ii][k])
                for ii in range(minik+1,maxik,1):
                    sum += max(0.0, R[ii][k])
                for ii in range(maxik+1,N,1):
                    sum += max(0.0, R[ii][k])
                A[i][k] = (1-lbda)*min(0.0, R[k][k]+sum) + lbda*A[i][k]
# find exemplar
E = np.array(R) + np.array(A)
idx = E.argmax(axis = 1)

print("Number of Cluster:",len(set(idx)))
print(set(idx))

# # OUTPUT
# for i in range(0,N,1):
#     print(idx[i]+1)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# plot nodes
plt.figure()
for ch_index, data in enumerate(locationPoint):
    head = locationPoint[idx[ch_index]]
    connection_x = [data[0], head[0]]
    connection_y = [data[1], head[1]]
    plt.plot(connection_x, connection_y, 'b', linestyle='-', marker='o', markersize=3)
plt.show()
