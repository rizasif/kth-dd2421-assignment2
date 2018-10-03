import numpy as np
from scipy.optimize  import minimize
import random, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def SVMClassifier(inputs, targets, C, N, PCM):

    def zerofun(alpha_,):
        #alpha_t=np.transpose(alpha_)
        return np.dot(alpha_, targets)

    def objective(alpha_):
        alphaProduct = -sum(alpha_)
        for i in range(N):
            for j in range(N):
                prod = 0.5 * alpha_[i] * alpha_[j] * PCM[i][j]
                alphaProduct += prod

        return alphaProduct

    bounds=[(0, C) for b in range(N)]
    constraint = {'type':'eq', 'fun':zerofun}
    alpha = minimize(objective, np.zeros(N), bounds=bounds,
     constraints=constraint).x
    return alpha

def generateData():
    np.random.seed(100)
    classA = np.concatenate((
    np.random.randn(10,2)*0.2+[1.5,0.5],
    np.random.randn(10,2)*0.2+[-1.5,0.5]))

    classB = np.random.randn(20,2)*0.5+[0.0,-0.5]

    #Plotting
    #pA = plt.plot([p[0] for p in classA], [p[1] for p in classA],'bo')
    #pB = plt.plot([p[0] for p in classB], [p[1] for p in classB],'ro')

    inputs = np.concatenate((classA,classB))
    targets = np.concatenate(
    (np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

    N = inputs.shape[0]

    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute,:]
    targets = targets[permute]

    return inputs, targets, N

def kernel(x1,x2,type='lin'):
    x1_t=np.transpose(x1)
    if type == 'lin':
        return np.dot(x1_t,x2)
    elif type == 'pol2':
        return math.pow(np.dot(x1_t,x2)+1,2)
    elif type == 'pol3':
        return math.pow(np.dot(x1_t,x2)+1,3)
    elif type == 'exp':
        return math.exp(-np.linalg.norm(x1-x2, 2)**2/(2.*2**2))


def preComputeMatrix(inputs, targets, N):
    M = []
    for i in range(N):
        A = []
        for j in range(N):
            k = kernel(inputs[i], inputs[j])
            A.append(k*targets[i]*targets[j])
        M.append(np.array(A))

    return np.array(M)

def extractZeroAlphas(data, inputs, targets, threshold):
    zeroPoints = []
    zeroTargets = []
    zeroAlpha = []
    goodPoints = []
    goodTargets = []
    goodAlpha = []
    for i in range(len(data)):
        if data[i]<threshold:
            zeroAlpha.append(data[i])
            zeroPoints.append(inputs[i])
            zeroTargets.append(targets[i])
        else:
            goodAlpha.append(data[i])
            goodPoints.append(inputs[i])
            goodTargets.append(targets[i])

    return goodAlpha, goodPoints, goodTargets, zeroAlpha, zeroPoints, zeroTargets

def bCalculation(alphas, inputs, targets, C):
    si = 0
    for i in range(len(alphas)):
        if alphas[i] < C:
            si = i
            break
    ans = 0
    for i in range(len(inputs)):
        ans += alphas[i]*targets[i]*kernel(inputs[si], inputs[i])
    return ans - targets[si]

def indicator(alphas, inputs, targets, b):
    ans = []
    for si in range(len(alphas)):
        sm = 0
        for i in range(len(alphas)):
            sm += alphas[i]*targets[i]*kernel(inputs[si],inputs[i])
        sm -= b
        ans.append(sm)

    return ans

def sIndicator(sv, alphas, inputs, targets, b):
    sm = 0
    for i in range(len(alphas)):
        sm += alphas[i]*targets[i]*kernel(sv,inputs[i])
    sm -= b
    return sm

if __name__ == "__main__":
    bestC, bestCF1, bestRes = None, None, None
    inputs, targets, N = generateData()
    preComputedMatrix = preComputeMatrix(inputs, targets, N)
    threshold = math.pow(10, -5)
    C = 5

    res = SVMClassifier(inputs, targets, C, N, preComputedMatrix)
    goodAlpha, goodPoints, goodTargets, zeroAlpha, zeroPoints, zeroTargets=\
    extractZeroAlphas(res, inputs, targets, threshold)
    b = bCalculation(goodAlpha, goodPoints, goodTargets, C)
    #ind =  indicator(goodAlpha, goodPoints, goodTargets, b)
    print ("SMV with C={}, res={}".format(C, goodAlpha))

    goodClassA=[]
    goodClassB=[]
    for i in range(len(zeroTargets)):
        if zeroTargets[i] == 1:
            goodClassA.append(zeroPoints[i])
        else:
            goodClassB.append(zeroPoints[i])
    pA = plt.plot([p[0] for p in goodClassA], [p[1] for p in goodClassA],'bo')
    pB = plt.plot([p[0] for p in goodClassB], [p[1] for p in goodClassB],'ro')



    xgrid = np.linspace(-5,5)
    ygrid = np.linspace(-4,4)
    grid = np.array([[sIndicator(np.array([x,y]),
    goodAlpha, goodPoints, goodTargets, C)
    for x in xgrid]
    for y in ygrid])

    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
    colors=('red', 'black', 'blue'),
    linewidths=(1,3,1))

    blue_patch = mpatches.Patch(color='blue', label='ClassA')
    red_patch = mpatches.Patch(color='red', label='ClassB')
    black_patch = mpatches.Patch(color='black', label='Decision Boundry')
    plt.legend(handles=[blue_patch, red_patch, black_patch])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
