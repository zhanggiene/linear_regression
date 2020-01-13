print(__doc__)



import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
import math

plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
#plt.show()

'''feature1=5
feature2=6'''
feature1=5
feature2=6
housingData=datasets.load_boston()
#use houseData.keys() to find out more
#dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

X=housingData['data']
#print(X)
y=housingData['target']
labels=housingData['feature_names']
#print(X.shape) it contains 506 examples of 13 features. 
indices = (feature1, feature2)

xLabel=labels[feature1]
yLabel=labels[feature2]
zLabel=labels[-1]
X_train = X[:-20, indices]  #extractedData = data[:,[1,9]]  select multiple colown without using ::
#it is selecting all the rows until the last 20
numExample=X_train.shape[0]

oneColoumn=np.full((numExample,1), 1)
#print(oneColoumn)
XInput=np.hstack([oneColoumn,X_train])
#print(XInput)
X_test = X[-20:, indices]
y_train = y[:-20]
y_test = y[-20:]


wleft=np.linalg.inv(np.matmul(XInput.transpose(), XInput))
wright=np.matmul(XInput.transpose(),y_train)
wVector=np.matmul(wleft,wright)

c=wVector[0]




# #############################################################################
# Plot the figure
recRange=np.max(X_train, axis=0)
#print(recRange)
xx, yy = np.meshgrid(range(math.ceil(recRange[0])), range(math.ceil(recRange[1])))  #x and y are the same thing 
#x[0,0] = 0    y[0,0] = 0
print(xx)

# there are 10*10 points in the grid, each point need x and y coordinate. 
#a plane is z=ax+by+c

zValue = wVector[1]*xx+wVector[2]*yy+c
#print(zValue)
# plot the surface


ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', marker='+') # color and marker. 
surface=ax.plot_surface(xx, yy, zValue,alpha=0.3)
#plotting the graph of colomn 1, colomn 2, 
plt.xlabel(xLabel)
plt.ylabel(yLabel)
#print([x for x in dir(surface) if 'set_' in x])

plt.show()
