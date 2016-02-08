from random import random
import sys
import numpy as np
import csv
from NaiveBayes import NBClassifier, make_sparse_data_rep, data_parser
training_f = sys.argv[1]
test_f = sys.argv[2]

def find_acc(pred_labels, labels):
	diffs= pred_labels+labels
	correct=0
	for diff in diffs:
		if diff!=0:
			correct+=1
	accuracy = correct/float(len(labels))
	return accuracy

class AdaBoost:
	
	def __init__(self, training_set, test_set):
		self.train_labels, self.train_data_instances = data_parser(training_set)
		self.test_labels, self.test_data_instances = data_parser(test_set)
		self.eval_labels, self.eval_data_instances = data_parser(test_set)
		self.numb_tr_data = len(self.train_labels)
		self.numb_t_data = len(self.test_labels)
		self.weights = 1/float(len(self.train_labels))*np.ones(len(self.train_labels))

	def sampled_data(self):
		accu_weight = [0]
		for i in range(1, len(self.weights)+1):
			accu_weight.append(accu_weight[i-1]+self.weights[i-1])
		data_indices = []
		for j in range(len(self.train_labels)):
			ran = random()
			for i in range(len(accu_weight)-1):
				if ran>=accu_weight[i] and ran<accu_weight[i+1]:
					data_indices.append(i)
		return data_indices


	def training(self):
		self.collection_classifier=[]
		self.alphas=[]
		epoch=0
		acc=0
		self.max_epoch=0
		final_values = np.zeros(len(self.train_labels))
		while epoch<100:
			data_indices = self.sampled_data()
			t_data = [self.train_data_instances[i] for i in data_indices]
			t_labels = [self.train_labels[i] for i in data_indices]
			bayes = NBClassifier((t_labels, t_data), ( self.train_labels, self.train_data_instances), (self.eval_labels, self.eval_data_instances))
			bayes.get_likelihood()
			pred = bayes.predict()
			tmp_labels = bayes.pred_train_labels
			diff_labels = tmp_labels+np.array(t_labels)
			err = 0.0
			correct_indices=[]
			for i in range(len(diff_labels)):
				if diff_labels[i]==0:
					err+=1
				else:
					correct_indices.append(i)
			error = err/float(len(t_labels))

			if error<0.5:
				alpha = np.log((1-error)/error)
				self.alphas.append(alpha)
				self.collection_classifier.append(bayes)
				epoch+=1
				for i in range(len(correct_indices)):
					self.weights[i]=self.weights[i]/np.log((1-error)/float(error))
				training_label = bayes.pred_test_labels
				final_values += alpha*training_label
				final_labels = np.sign(final_values)
			accuracy = find_acc(final_labels, self.train_labels)
			if accuracy>acc:
				acc=accuracy
				self.max_epoch = epoch
		

	def predict(self):
		final_values_test = np.zeros(len(self.eval_labels))
		final_values_train = np.zeros(len(self.train_labels))
		for i in range(self.max_epoch):
			alpha = self.alphas[i]
			classifier = self.collection_classifier[i]
			test_values = classifier.pred_eval_labels
			train_values = classifier.pred_test_labels
			final_values_train += alpha*train_values
			final_values_test += alpha*test_values
		final_labels_train = np.sign(final_values_train)
		final_labels_test = np.sign(final_values_test)
		test_diff = 2*final_labels_test+self.eval_labels
		train_diff = 2*final_labels_train+self.train_labels
		ttp=0; tfn=0; tfp=0; ttn=0; trtp=0; trfn=0; trfp=0; trtn=0;
		for x in test_diff:
			if x==3:
				ttp+=1
			elif x==-1:
				tfn+=1
			elif x==1:
				tfp+=1
			elif x==-3:
				ttn+=1
		for y in train_diff:
			if y==3:
				trtp+=1
			elif y==-1:
				trfn+=1
			elif y==1:
				trfp+=1
			elif y==-3:
				trtn+=1
		return ttp, tfn, tfp, ttn, trtp, trfn, trfp, trtn 


def compute_eval_matrics(tp, fn, fp, tn):
	precision = tp/float(tp+fp)
	recall = tp/float(tp+fn)
	f=2*precision*recall/float(precision+recall)
	f_05 = (1+0.5**2)*precision*recall/float((0.5**2)*precision+recall)
	f_2 = (1+2**2)*precision*recall/float((2**2)*precision+recall)
	sensitivity=tp/float(tp+fn)
	specificity=tn/float(fp+tn)
	accuracy = (tp+tn)/float(tp+fn+fp+tn)
	error=1-accuracy
	return accuracy, error, sensitivity, specificity, precision, f, f_05, f_2

def main():
	a = AdaBoost(training_f, test_f)
	a.training()
	ttp, tfn, tfp, ttn, trtp, trfn, trfp, trtn = a.predict()
	print trtp,' ', trfn, ' ', trfp, ' ', trtn,'\n',ttp,' ', tfn, ' ', tfp, ' ', ttn
	# #for train
	# accuracy, error, sensitivity, specificity, precision, f, f_05, f_2 = compute_eval_matrics(trtp, trfn, trfp, trtn)
	# print accuracy, error, sensitivity, specificity, precision, f, f_05, f_2
	# #for test
	# accuracy, error, sensitivity, specificity, precision, f, f_05, f_2 = compute_eval_matrics(ttp, tfn, tfp, ttn)
	# print accuracy, error, sensitivity, specificity, precision, f, f_05, f_2

if __name__=='__main__':
	main()


