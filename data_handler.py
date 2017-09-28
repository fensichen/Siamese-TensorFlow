#data_handler.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy.random 
import itertools

class DataHandler:
	def __init__(self):
		self.mnist = input_data.read_data_sets("MNIST_data")

	def get_mnist(self):
		return self.mnist

	# 
	def preprocess(self, images, labels):
		
		self.labels  = labels		
		
		images       = images.reshape((55000, 28, 28))
		self.images  = np.zeros((55000, 28, 28, 3))
		# make fake channel
		for i in range ( 55000 ):
			img              = images[i]
			img              = np.reshape(img, [img.shape[0], img.shape[1], -1])
			img_tmp          = np.zeros( (img.shape[0], img.shape[1], 3 ))
			img_tmp[:,:,0:1] = img
			img_tmp[:,:,1:2] = img
			img_tmp[:,:,2:3] = img
			img              = img_tmp

			self.images[i]   = img	 


		self.num_idx = dict()
		for idx, num in enumerate ( self.labels ):
			if num in self.num_idx:
				self.num_idx[num].append(idx)
			else:
				self.num_idx[num] = [idx]

		self.to_img = lambda x: self.images[x]

	def next_batch(self, batch_size):
		left        = []
		right       = []
		sim         = []
		for i in range(10):
			n = 45
			l = np.random.choice( self.num_idx[i], 2*n, replace=False).tolist() # whether the sample is without replacement
			left.append(self.to_img(l.pop()))
			right.append(self.to_img(l.pop()))
			sim.append([1])

		
		# return r = 2 length subsequences of elements from the input itera
		for i,j in itertools.combinations(range(10),2):
			left.append( self.to_img( np.random.choice(self.num_idx[i])))
			right.append(self.to_img( np.random.choice(self.num_idx[j])))
			sim.append([0])

		left        = np.array(left)
		right       = np.array(right)
		sim         = np.array(sim)


		return left, right, sim

if __name__ == '__main__':

	datahandler = DataHandler()
	data        = datahandler.get_mnist()
	train 		= data.train.images
	label 		= data.train.labels

	datahandler.preprocess( train, label )
	datahandler.next_batch(10)