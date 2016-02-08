import sys
import numpy as np
import csv
training_f = sys.argv[1]
test_f = sys.argv[2]
#this function acts on each data point and separate the label and the dictionary of attributes out from eact other.
def make_sparse_data_rep(data_pt):
	label = int(data_pt[0])
	tmp_data = data_pt[1:len(data_pt)]
	sparse_rep = [] #initialize a sparse representation dict for the matrix
	for i in range(len(tmp_data)):
		stuff = tmp_data[i]
		sparse_rep.append(stuff)
	return label, sparse_rep


def data_parser(filename):
	#writing a function to parse a LIBSVM data
	i_f = open(filename, 'r')
	reader = csv.reader(i_f, delimiter=' ')
	data_list = []
	for line in reader:
		data_list.append(line)
	label_vec = []
	data_instance = []
	for data in data_list:
		label, sparse_rep = make_sparse_data_rep(data)
		label_vec.append(label)
		data_instance.append(sparse_rep)
	return label_vec, data_instance

class NBClassifier:

	def __init__(self, training_set, test_set, eval_set):
		self.train_labels, self.train_data_instances = training_set
		self.test_labels, self.test_data_instances = test_set
		self.eval_labels, self.eval_data_instances = eval_set
		self.numb_tr_data = len(self.train_labels)
		self.numb_t_data = len(self.test_labels)

	def get_priors(self):
		self.number_pos = 0
		self.number_neg = 0
		pos_index = []
		neg_index = []
		for i in range(len(self.train_labels)):
			if self.train_labels[i] == 1:
				self.number_pos+=1
				pos_index.append(i)
			else:
				self.number_neg+=1
				neg_index.append(i)
		self.prior_pos = self.number_pos/float(self.numb_tr_data)
		self.prior_neg = self.number_neg/float(self.numb_tr_data)
		return pos_index, neg_index

	def get_likelihood(self):
		pos_dat_index, neg_dat_index = self.get_priors()
		pos_data = [self.train_data_instances[i] for i in pos_dat_index]
		neg_data = [self.train_data_instances[i] for i in neg_dat_index]
		freq_dict_pos = {}
		freq_dict_neg = {}
		for pos_datum in pos_data:
			for attr in pos_datum:
				if attr in freq_dict_pos:
					freq_dict_pos[attr]+=1
				else:
					freq_dict_pos[attr] = 1
		for neg_datum in neg_data:
			for attr in neg_datum:
				if attr in freq_dict_neg:
					freq_dict_neg[attr]+=1
				else:
					freq_dict_neg[attr] = 1
		self.likelihood_pos = {i:freq_dict_pos[i]/float(self.number_pos) for i in freq_dict_pos.keys()}
		self.likelihood_neg = {i:freq_dict_neg[i]/float(self.number_neg) for i in freq_dict_neg.keys()}


	def predict(self):
		self.pred_test_labels= np.array([self.naive_bayes(test_datum) for test_datum in self.test_data_instances])
		self.pred_train_labels = np.array([self.naive_bayes(train_datum) for train_datum in self.train_data_instances])
		self.pred_eval_labels = np.array([self.naive_bayes(eval_datum) for eval_datum in self.eval_data_instances])
		test_diff = 2*self.pred_test_labels+self.test_labels
		train_diff = 2*self.pred_train_labels+self.train_labels
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


	def naive_bayes(self, datum):
		likelihood_ifpos = self.prior_pos
		likelihood_ifneg = self.prior_neg
		for attr in datum:
			if attr in self.likelihood_pos:
				likelihood_ifpos *= self.likelihood_pos[attr]
			else:
				likelihood_ifpos *= 0
			if attr in self.likelihood_neg:
				likelihood_ifneg *= self.likelihood_neg[attr]
			else:
				likelihood_ifneg *= 0
		if likelihood_ifpos>likelihood_ifneg:
			return 1
		else:
			return -1

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
	training = data_parser(training_f)
	testing = data_parser(test_f)
	a = NBClassifier(training, testing, testing)
	a.get_likelihood()
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


