import numpy as np
import random

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-6
    i = 0
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished and i<72:
        ix = it.multi_index
        #print 'ix start again',ix, x[ix]
        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later
        ### YOUR CODE HERE:
        # we check coordinates per coordinates
        random.setstate(rndstate)
        #define a vector of zeros with only one perturbation for one coordinates
        U_ix = np.zeros(x.shape)
        U_ix[ix] = h
        fx = f(x)[0]
        #print "fx",fx
        random.setstate(rndstate)
        #compute the function f for this pertubated vector
        #print "y", (x+U_ix)[(0,)]
        #print "x",x[(0,)]
        fy = f(x+U_ix)[0]
        #compute the gradient
        #print "fy",fy
        numgrad = (fy-fx)/h
        ### END YOUR CODE
        # Compare gradients
        reldiff = np.max(np.abs(numgrad - grad[ix])) / max(1, np.max(np.abs(numgrad)), np.max(np.abs(grad[ix]))) #you switch to np.max and abs 
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    #gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    #sanity_check(f_gradf(np.array[1.,2.,3.]),np.array[1.,2.,3.])
    sanity_check()
    your_sanity_checks()
    

