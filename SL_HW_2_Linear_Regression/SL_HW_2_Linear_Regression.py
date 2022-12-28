# * write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.? hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html) to be more efficient.

class Ols(object):
    def __init__(self):
        self.w = None
    
    @staticmethod
    def pad(X):
        return np.hstack((np.ones((X.shape[0],1)), X))
  
    def fit(self, X, Y):    
        self.w = (np.linalg.pinv(self.pad(X)).dot(Y))
    
    def predict(self, X):
        return self.pad(X).dot(self.w)
    
    def score(self, X, Y): # returns the MSE on a given sample set  
        return ((Y - self.predict(X))**2).sum() / Y.shape[0]


# Write a new class OlsGd which solves the problem using gradinet descent. 
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
        



class Normalizer():
    """The function performs Z - normalization"""
    def __init__(self):
        pass

    def fit(self, X):
        self.means = X.mean(axis=0)
        self.std = np.sqrt(((X - self.means)**2).sum(axis=0) / X.shape[0])

    def predict(self, X):
    #apply normalization
        return (X - self.means) / self.std
    



class OlsGd(Ols):
  
    def __init__(self, learning_rate=.05, 
               num_iteration=1000, 
               normalize=True,
               early_stop=True,
               early_stop_delta=0.001,
               verbose=True):
    
        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer = Normalizer()    
        self.verbose = verbose
        self.early_stop_delta = early_stop_delta
    
    def _fit(self, X, Y, reset=True, track_loss=True):
    #remeber to normalize the data before starting
        if self.normalize:
            self.normalizer.fit(X)
            X = self.normalizer.predict(X)
    
        if track_loss: # Initialization of loss tracker
            self.loss_tracker = [] 
    
        self.w = np.ones(self.pad(X).shape[1]) # Initial weights
        self.y_pred = self._predict(X)
    
        if self.early_stop: # Initialization of early stopping
            prev_loss = np.inf
            delta = np.inf
            i = 0
        
            while delta > self.early_stop_delta:
                self._step(X, Y)
                i += 1
                delta = prev_loss - self.loss
                prev_loss = self.loss
                try:
                    self.loss_tracker.append(self.loss)
                except: continue
                if self.verbose == 2:
                    print(f'Epoch: {i}, Loss: {self.loss}')
          
            else: 
                if self.verbose == 1:
                    print(f'Early stop was triggered at {i}-th epoch with loss: {self.loss}')
            
        else:
            for i in range(self.num_iteration):
                self._step(X, Y)
                try:
                    self.loss_tracker.append(self.loss)
                except: continue
                if self.verbose == 2:
                    print(f'Epoch: {i}, Loss: {self.loss}')

    def _predict(self, X):
        #remeber to normalize the data before starting
        if self.normalize:
            X = self.normalizer.predict(X)
        return self.predict(X)

    def _step(self, X, Y):
        # use w update for gradient descent
        self.loss = (self.y_pred - Y).T.dot(self.y_pred - Y) / 2 # Loss function from the lecture slides 
        d_loss = (self.pad(X).T.dot(self.y_pred - Y)) # Loss derivative
        self.w = self.w - self.learning_rate * (1/X.shape[0]) * d_loss # Update weights
        self.y_pred = self.predict(X)




class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda
    
    def _fit(self, X, Y):
        self.w = (np.linalg.inv(self.pad(X).T.dot(self.pad(X)) 
                            + self.ridge_lambda * np.identity(self.pad(X).shape[1])).dot(self.pad(X).T).dot(Y))
    



class RidgeLsGd(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLsGd,self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _step(self, X, Y):
        # use w update for gradient descent
        self.loss = (self.y_pred - Y).T.dot(self.y_pred - Y) / 2 + self.ridge_lambda * np.linalg.norm(self.w)# Loss function from the lecture slides 
        d_loss = (self.pad(X).T.dot(self.y_pred - Y)) + self.ridge_lambda * self.w # Loss derivative
        self.w = self.w - self.learning_rate * (1/X.shape[0]) * d_loss # Update weights
        self.y_pred = self.predict(X)
    