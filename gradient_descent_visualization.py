#https://towardsdatascience.com/implementation-of-multi-variate-linear-regression-in-python-using-gradient-descent-optimization-b02f386425b9
#plot animation https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from sklearn import datasets
from sklearn import preprocessing
import math

def hypothesis(theta, X):
    #x is the actual training data 
    f=X.shape[1]   #amount of features
    m=X.shape[0]   #amount of data
    theta = theta.reshape(f+1,1)  # need to plus one because of the b, need to account for it. 
    #theta is passed in as numpy array, but we need to shape into coloum vector. 
    h = np.ones((m,1)) # create a coloumn of 1. 
    XInput=np.hstack([h,X])
    result=np.matmul(XInput,theta)
    #basically calculating y=b+x1*theta1+x2*theta2+x3*theta3.......... 
    return result.reshape(m) 
    # return type is also numpy array. 
def gradient_descent(X,y,theta,learning_rate=1e-25,iterations=100):
   
    # X is training data without adding 1 in front
    # y is the training result, numpy array, not coloum array
    #theta is the parameter
    m=X.shape[0]
    f=X.shape[1]# amount of features

    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,f+1))
    for it in range(iterations):

        result=hypothesis(theta,X) #this result is stored in flattened numpy array
        #theta is constantly changing

        cost_history[it]=(1/m) * 0.5 * sum(np.square(result - y))
        
        # differentiate wrt b 
        theta[0] = theta[0] -(1/m)*learning_rate*(sum(result-y)) 

        #differentiate wrt w1,w1,w3,w4......
        for j in range(1,f+1):
            theta[j] = theta[j] - (1/m)*learning_rate*sum((result-y) * X.T[j-1])    # the X here dont have 1 coloumn in front
        theta_history[it] =theta
        

        
    return theta, cost_history, theta_history
    #return theta,cost_history



fig=plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 2, 1, projection='3d')

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
X_test = X[-20:, indices]
y_train = y[:-20]
y_test = y[-20:]


#print(preprocessing.scale(X_train))   # scale the dataset to be -1 -1. with standard diviation of zero. 
X_train_scale=preprocessing.scale(X_train)
Y_train_scale=preprocessing.scale(y_train)


print("")

'''
mean = np.ones(X_train.shape[1])
std = np.ones(X_train.shape[1])
for i in range(0, X_train.shape[1]):
    mean[i] = np.mean(X_train.transpose()[i])
    std[i] = np.std(X_train.transpose()[i])
    for j in range(0, X_train.shape[0]):
        X_train[j][i] = (X_train[j][i] - mean[i])/std[i]
print(X_train)'''




######################draw graph
recRange=np.max(X_train, axis=0)

#xx, yy = np.meshgrid(range(math.ceil(recRange[0])), range(math.ceil(recRange[1])))  #x and y are the same thing 
#span of plan is -2 to 2 as i have normalize the data to be -2 to 2. 
xx, yy = np.meshgrid(np.linspace(-2, 2, 100) , np.linspace(-2, 2, 100))  #x and y are the same thing 

#https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
theta = np.zeros(X_train.shape[1]+1)

''' change the value of initial state of theta here, '''

theta[0]=-2
theta[1]=-2
theta[2]=-2

final_theta,cost_history,theta_history=gradient_descent(X_train_scale,Y_train_scale,theta,learning_rate=0.01,iterations=500)

#y=ax+by+c

#zarray[i] has to be in form of [100,100] as it has to be the same as xx,yy (check meshgrid to know why. )
zarray = np.zeros((500, 100,100))
for i in range(500):
    zarray[i] = theta_history[i][1]*xx+theta_history[i][2]*yy+theta_history[i][0]
    #print(theta_history[i][1])
ax.set_xlim(-4,4)
ax.set_ylim(-2,2)
ax.set_zlim(-2,4)
ax.scatter(X_train_scale[:, 0], X_train_scale[:, 1], Y_train_scale, c='r', marker='+') # color and marker. 
plot=[ax.plot_surface(xx, yy, zarray[0],alpha=0.3)]  # have no idea why need to put in [],




def update(ifrm,zarray,plot):
    ax.set_title('Frame ' + str(ifrm))
    plot[0].remove()   #dont really understand, u cannot change to ax.remove. 
    #plot.remove()
    plot[0] =ax.plot_surface(xx, yy, zarray[ifrm],alpha=0.3)
#y=ax+by+c  we are not just calculating one value . xx,yy,zarray has to be in form of [100][100] numpy array so plane can be drawn 
    #return plot
ani = animation.FuncAnimation(fig, update, 500, fargs=(zarray,plot),interval=1)
#plotting the graph of colomn 1, colomn 2, 
plt.xlabel(xLabel)
plt.ylabel(yLabel)

print(cost_history)




plt.show()
