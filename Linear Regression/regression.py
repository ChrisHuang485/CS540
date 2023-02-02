import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    dataset = {}
    dataset = genfromtxt(filename, delimiter=',')
    dataset = np.delete(dataset,0,0)
    dataset = np.delete(dataset,0,1)
    return dataset


def print_stats(dataset, col):
    num = 0
    total = 0.00
    avg = 0.00
    sd = 0.00
    std = 0.00
    x = 0
    for i in dataset[:,col]:
        if dataset[:,col] is not None:
            num += 1
            total += dataset[x,col]
            x+=1
            
    avg = round(total/num,2)
    y = 0
    for i in dataset[:,col]:
        sd += (dataset[y,col] - avg) ** 2
        std =  round(np.sqrt((1/(num-1)) * sd),2)
        y += 1
        
    print(num)
    print(avg)
    print(std)

def regression(dataset, cols, betas):
    num = 0
    for i in dataset[:,cols]:
        if dataset[:,cols] is not None:
            num += 1
    new_betas = []
    for i in range(len(cols)):
        new_betas.append(betas[i+1])
        
    X = np.dot(dataset[:,cols],new_betas)
    Y = dataset[:,0]
    data = 1/num * (betas[0] + X-Y) ** 2
    mse = np.sum(data)
    return mse


def gradient_descent(dataset, cols, betas):
    num = 0
    grads = []
    for i in dataset[:,cols]:
        if dataset[:,cols] is not None:
            num += 1
            
    new_betas = []
    for i in range(len(cols)):
        new_betas.append(betas[i+1])
        
    X = np.dot(dataset[:,cols],new_betas)        
    Y = dataset[:,0]
    data = 2/num  * (betas[0] + X-Y)
    data0 = np.sum(data)
    grads.append(data0)
    for i in range(len(cols)):
        grads.append(np.sum(2/num  * (betas[0] + X-Y) * dataset[:,cols[i]]))
    grads = np.array(grads)
    return grads
    

def iterate_gradient(dataset, cols, betas, T, eta):
    output = []

    for num in range(1,T+1):
        new_betas = []
        for i in range(len(betas)):
            new_betas.append(betas[i] - eta * gradient_descent(dataset, cols, betas)[i])
        mse = regression(dataset, cols,new_betas)
        print(num,end = " ")
        print('%.2f'%mse,end = " ")
        for n in range(len(betas)):
            print('%.2f'%(betas[n] - eta * gradient_descent(dataset, cols, betas)[n]), end = " ")
        print()
        betas = new_betas



def compute_betas(dataset, cols):
    num = 0
    for i in dataset[:,cols]:
        if dataset[:,cols] is not None:
            num += 1
    data_1 = [[1]] * num
    data = dataset[:,cols]
    X = np.c_[data_1,data]
    Y = dataset[:,0]

    betas = None
    betas =  np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    mse = regression(dataset, cols,betas)
    return (mse, *betas)


def predict(dataset, cols, features):

    betas = compute_betas(dataset, cols)[1:]
    result = betas[0]
    for i in range(len(features)):
        result += betas[i + 1] * features[i]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    result1 = []
    result2 = []
    for i in range(len(X)):
        y1 = betas[0] + np.dot(betas[1] , X[i,0]) + np.random.normal(0,sigma)
        array1 = [y1,X[i,0]]
        result1.append(array1)

    for i in range(len(X)):
        y2 = alphas[0] + np.dot(alphas[1] , X[i,0] ** 2) + np.random.normal(0,sigma)
        result2.append([y2,X[i,0]])
    return np.array(result1), np.array(result2)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    X = np.transpose(np.array([np.random.randint(-100,100,1000)]))
    betas = np.array([1,2])
    alphas = np.array([2,3])
    sigma = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
    data = []
    result1 = []
    result2 = []
    for i in range(len(sigma)):
        data = synthetic_datasets(betas, alphas, X, sigma[i])
        mse1 = compute_betas(data[0],[1])[0]
        mse2 = compute_betas(data[1],[1])[0]
        result1.append(mse1)
        result2.append(mse2)
        
    plt.plot(sigma, result1, '-o', label = "MSE of Linear Dataset")
    plt.plot(sigma, result2, '-o', label = "MSE of Quadratic Dataset")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Standard Deviation of Error Term")
    plt.ylabel("MSE of Trained Model")
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
    
