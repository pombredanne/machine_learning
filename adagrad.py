import numpy as np


#Adagrad
class Adagrad:
    
    def __init__(self, training_dat, test_dat):
        self.training_dat = training_dat
        self.test_dat = test_dat
        self.numb_data = self.training_dat.shape[0]
        self.numb_test_data = self.test_dat.shape[0]
        self.numb_features = self.training_dat.shape[1]-1
        self.weight = np.zeros(self.numb_features+1)#initialize the weigth vector to be zero for all n+1 features
        self._setup_()
    
    def _setup_(self):
        intercept_vector = np.ones(self.numb_data) # initialize the intercept vector
        intercept_vector_t = np.ones(self.numb_test_data) #initialize the intercept vector for test
        augmented_test_sample = np.column_stack((intercept_vector_t,self.test_dat))
        augmented_sample = np.column_stack((intercept_vector, self.training_dat))# incoroperate the intercept vector into the feature vectors 
        self.x_test = np.transpose(np.transpose(augmented_test_sample)[0:self.numb_features+1])
        self.y_test = np.transpose(augmented_test_sample)[self.numb_features+1]
        self.x_sample = np.transpose(np.transpose(augmented_sample)[0:self.numb_features+1]) #extracting the mx(n-1) data matrix
        self.y_label = np.transpose(augmented_sample)[self.numb_features+1] #extracting the mx1 label vector
        self.g_square = np.empty(self.numb_features+1) 
        self.g_square.fill(1*10**(-23))
    
    def update(self, rate, i):
        raw_y = np.dot(self.weight, self.x_sample[i])
        if self.y_label[i]*raw_y <= 1:
            gradient = -(self.y_label[i]*self.x_sample[i])
            self.g_square += gradient**2
            rms_g = np.sqrt(self.g_square)
            tmp_grad = (gradient)*(rms_g**-1)   
            self.weight += -1*(rate*tmp_grad)
        if self.y_label[i]*raw_y<=0:
            return 1
        else:
            return 0
        

    def prediction(self):
        mistakes = 0
        for i in range(self.numb_test_data):
            raw_y_test = np.dot(self.weight,self.x_test[i])
            if self.y_test[i]*raw_y_test<=0:
                mistakes+=1
        number_correct = self.numb_test_data-mistakes
        return (number_correct/float(self.numb_test_data))*100
            
