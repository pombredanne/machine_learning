import numpy as np
import math

#Perceptron
class ModPerceptron:
    
    def __init__(self, training_dat, test_dat, margin = False):
        self.training_dat = training_dat
        self.test_dat = test_dat 
        self.margin = margin
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
        
    
    def update(self, rate_tup, i, gamma=1.0):
        eta1, eta2 = rate_tup
        raw_y = np.dot(self.weight, self.x_sample[i])
        if self.y_label[i] == -1:
            if self.y_label[i]*raw_y <= 0:
                self.weight +=eta1*self.y_label[i]*self.x_sample[i]
                return 1
            else:
                return 0
        elif self.y_label[i] == 1:
            if self.y_label[i]*raw_y <= 0:
                self.weight +=eta2*self.y_label[i]*self.x_sample[i]
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