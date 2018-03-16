def featureNormalization(X):
    """ Feature Normalization """
    # my code here
    normalized_X = (X - np.average(X)) / np.std(X)
    return normalized_X

def normalEquation(X, y):
    # Normal Equations
    trans = np.transpose(X)
    transDot = np.dot(trans, X)
    inversed = np.linalg.inv(transDot)
    theta = np.dot((np.dot(inversed, trans)), y)
    return theta

def computeCost(X, y, theta=[[0],[0]]):
    """ Computing Cost (for Multiple Variables) """
    J = 0
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m)) * np.sum(np.square(h-y))
    
    return(J)

def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    """ Gradient Descent (for Multiple Variables) """
    #     J_history array of cost finction values per iteration
    m = y.size
    J_history = []
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1/m)*(X.T.dot(h-y))
        J_history.append(computeCost(X, y, theta))
    return(theta, J_history)

def sigmoid(z):
    # my code here
    t = pow(np.e, -z)
    s = pow(1 + t, -1)
    return s

def costFunction(theta, X, y):
    # my code here
    m = y.size
    a1 = np.dot(X, theta)
    a2 = sigmoid(a1)
    a3 = np.log(np.transpose(a2))
    a4 = np.dot(a3, y)
    b1 = 1 - a2
    b2 = np.transpose(b1)
    b3 = np.log(b2)
    b5 = 1 - y
    b4 = np.dot(b3, b5)
    c1 = a4 + b4
    J = c1 / m
    return(-J[0])

def gradient(theta, X, y):
    # your code here
    m = y.size
    transposed = np.transpose(X)
    mult1 = X.dot(theta)
    sigmoidValue = sigmoid(mult1)
    matrix2 = sigmoidValue - y.flatten()
    grad = transposed.dot(matrix2) / m
    flattened = grad.flatten()
    return grad

# returns 1 or 0
def predict(theta, X, threshold=0.5):
    """ Logistic Regretion predict """
    X = sigmoid(X.dot(theta))
    return (X >= threshold).astype(int)

def costFunctionReg(theta, reg, *args):
    X = args[0]
    y = args[1]
    cost = costFunction(theta, X, y)
    m = y.size
    theta[0] = 0
    J = cost + (reg/(2*m)) * np.sum(np.square(theta))
    
    return(J)

def gradientReg(theta, reg, *args):
    # My code here
    X = args[0]
    y = args[1]
    oldGrad = gradient(theta, X, y)
    m = y.size
    theta[0] = 0
    grad = oldGrad + (reg / m) * theta
    return(grad)
