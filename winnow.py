import numpy as np
import math

class Winnow:
    def __init__(self, training_dat, test_dat, margin = False):
        self.training_dat = training_dat
        self.test_dat = test_dat
        self.margin = margin 
        self._setup_()
    
    def _setup_(self):
        self.numb_data = self.training_dat.shape[0]
        self.numb_test_data = self.test_dat.shape[0]
        self.numb_features = self.training_dat.shape[1]-1
        self.x_sample = np.transpose(np.transpose(self.training_dat)[0:self.numb_features]) #extracting the mx(n-1) data matrix
        self.y_label = np.transpose(self.training_dat)[self.numb_features] #extracting the mx1 label vector
        self.x_test = np.transpose(np.transpose(self.test_dat)[0:self.numb_features]) #extracting the mx(n-1) data matrix
        self.y_test = np.transpose(self.test_dat)[self.numb_features] #extracting the mx1 label vector
        self.weight = np.ones(self.numb_features)#initialize the weigth vector to be one for all n features
        self.theta = -self.numb_features
    
    def update(self, alpha, i, gamma=1.0):
        raw_y = np.dot(self.weight, self.x_sample[i])+self.theta
        if self.margin == True:
            if self.y_label[i]*raw_y<= gamma:
                self.weight = self.weight*(alpha**(self.y_label[i]*self.x_sample[i]))
            if self.y_label[i]*raw_y<=0:
                return 1
            else:
                return 0
        else:
            if self.y_label[i]*raw_y<=0:
                self.weight = self.weight*(alpha**(self.y_label[i]*self.x_sample[i]))
                return 1
            else:
                return 0
           

    
    def prediction(self):
        mistakes = 0
        for i in range(self.numb_test_data):
            raw_y_test = np.dot(self.weight,self.x_test[i])+self.theta
            if self.y_test[i]*raw_y_test<=0:
                mistakes+=1
        number_correct = self.numb_test_data-mistakes
        return (number_correct/float(self.numb_test_data))*100        
