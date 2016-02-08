import numpy as np
training_set=np.array([[1,1],[1.5,2],[2.5,5.5],[6,5],[4,5],[4.5,5],[3.5,4.5],[5,1],[6,1]])
centroids1=[(4,5),(3.5, 4.5),(6,5)]
centroids2=[(2.5,5.5),(3.5,4.5),(6,1)]

#calculate Eucidean distance.
def distance(x,y):
	diff = x-y
	return np.sqrt(np.dot(diff, diff))

#label cluster
def label_cluster(centroids, data_set):
	cluster_label_list=[]
	for data in data_set:
		sim_list=[]
		sim_dict={}
		for i in range(len(centroids)):
			similarity = distance(data, centroids[i])
			sim_list.append(similarity)
			sim_dict[similarity] = i
		cluster_label = sim_dict[min(sim_list)]+1
		cluster_label_list.append(cluster_label)
	return cluster_label_list

def compute_centroids(centroids, data_set, label_clusters):
	clusters = label_clusters
	number_of_centroids = len(centroids)
	number_of_data = len(training_set)
	for i in range(number_of_centroids):
		c=0
		sum=0
		for j in range(number_of_data):
			if clusters[j]-1==i:
				sum+=training_set[j]
				c+=1
		if c!=0:
			centroids[i]=sum/float(c)
	return centroids

def k_mean(init_centroids, data_set):
	labels = label_cluster(init_centroids, data_set) #label data based on the initial centroids
	old_centroid = init_centroids
	new_centroids = None
	while new_centroids!=[]:
		old_labels = labels
		new_centroids = compute_centroids(old_centroid, data_set, labels) #compute new centroids
		old_centroid = new_centroids
		labels = label_cluster(new_centroids, data_set)
		# print 'the labels are', labels
		# print 'the old labels are', old_labels
		if labels==old_labels:
			return labels



	




l = label_cluster(centroids2, training_set)
print k_mean(centroids1, training_set)
print k_mean(centroids1, training_set)
n_cen=compute_centroids(centroids2, training_set,l )
l2 = label_cluster(n_cen, training_set)
print compute_centroids(n_cen, training_set, l2)
print l
print l2




