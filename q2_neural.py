import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    #params corresponds to the orignial parameters contained in a single arrat
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H)) #first Dx+H parameters for the layers_0-layer_1 connection
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H)) # then  Dx+H to Dx +2H parameters for the layers_0-layer_1 bias
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    ### YOUR CODE HERE: forward propagation
    N = data.shape[0]
    h = sigmoid(np.dot(data,W1)+b1) # (N,H) i.e one row per observation, b1 is the bias  
    #print "h",h.shape
    ypred = softmax(np.dot(h,W2)+b2)  # (N,Dy) the softmax is applied by row
    #print "ypred", ypred.shape
    #print "labels", labels[n,:].reshape(1,labels[n,:].shape[0]).shape
    #print "cost",cost[n,:].shape
    cost = np.sum(np.multiply(labels,np.log(ypred)))#,axis = 1)
    #print "cost",cost.shape
    ### END YOUR CODE
    
    #compute gradient of each weights of layer1-layer2 connections : in an array of ..
    err2 = -(ypred-labels) #(N,Dy)
    #print "err2", err2.shape
    gradW2 = np.dot(err2.T,h).T   #(H,Dy) it includes the partial derivatives for each weight of the 2 last layers connections
    #print "shape gradW2",gradW2.shape
    gradb2 = np.sum(err2,axis=0) #(1,Dy)
    #print "shape gradb2",gradb2.shape                
    err1= np.multiply(np.dot(err2,W2.T),sigmoid_grad(h))  # (N,H)
    #print "err_1n",err1.shape        
    gradW1 = np.dot(data.T,err1)  #(Dx,N) #column index corresponds to the hidden layer connection and row index to                  
    #print "shape gradW1",gradW1.shape
    #print "gradW2",gradW2[0,0]
    gradb1 = np.sum(err1,axis=0) # (1,H)
    #print "shape gradb1",gradb1.shape
    ### YOUR CODE HERE: backward propagation
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    #print "grad", grad.shape
    return cost, grad
    
def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 12]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    
    
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)
    

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()