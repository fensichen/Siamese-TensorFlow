
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_handler


"""
Args: 
    inputs: a tensor of size [batch_size, height, width, channels]

"""

class Siamese(object):
    def __init__(self, vgg_16_model_path, margin, learning_rate):
        self.dict          = np.load(vgg_16_model_path, encoding="latin1").item()
        self.left          = tf.placeholder(tf.float32, [55, 28, 28, 3], name="left")
        self.right         = tf.placeholder(tf.float32, [55, 28, 28, 3], name="right")
        self.y             = tf.placeholder(tf.float32, [55, 1 ]       , name="label")
        self.margin        = margin
        self.learning_rate = learning_rate

        

    def conv2d(self, bottom, ksize, name, reuse):
        with tf.variable_scope(name) as scope:
            if reuse == True:
                scope.reuse_variables()

            shape    = bottom.get_shape().as_list()
            num_c    = shape[-1]
            kernel   = [ ksize[0], ksize[1], num_c, ksize[2] ]
            
            if name in self.dict.keys():
                init = tf.constant_initializer(value=self.dict[name][0], dtype=tf.float32)
                shape= self.dict[name][0].shape
            else:
                init = tf.contrib.layers.xavier_initialier_conv2d()

            
            w        = tf.get_variable(name + "/weights", shape = kernel,     initializer= init)
            b        = tf.get_variable(name + "/biases",  shape = [ksize[2]], initializer= tf.zeros_initializer()) 

        return tf.nn.conv2d( bottom, w, strides = [1,1,1,1], padding = 'SAME' ) + b 

    def network(self, X, reuse):
      
        conv1_1 = tf.nn.relu( self.conv2d(X,       [3,3,64], 'conv1_1', reuse) )
        conv1_2 = tf.nn.relu( self.conv2d(conv1_1, [3,3,64], 'conv1_2', reuse) )
        pool1   = tf.nn.max_pool( conv1_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv2_1 = tf.nn.relu( self.conv2d(pool1,   [3,3,128], 'conv2_1', reuse) )
        conv2_2 = tf.nn.relu( self.conv2d(conv2_1, [3,3,128], 'conv2_2', reuse) )
        pool2   = tf.nn.max_pool( conv2_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv3_1 = tf.nn.relu( self.conv2d(pool2,   [3,3,256], 'conv3_1', reuse) )
        conv3_2 = tf.nn.relu( self.conv2d(conv3_1, [3,3,256], 'conv3_2', reuse) )
        conv3_3 = tf.nn.relu( self.conv2d(conv3_2, [3,3,256], 'conv3_3', reuse) )
        pool3   = tf.nn.max_pool( conv3_3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv4_1 = tf.nn.relu( self.conv2d(pool3,   [3,3,512], 'conv4_1', reuse) )
        conv4_2 = tf.nn.relu( self.conv2d(conv4_1, [3,3,512], 'conv4_2', reuse) )
        conv4_3 = tf.nn.relu( self.conv2d(conv4_2, [3,3,512], 'conv4_3', reuse) )
        pool4   = tf.nn.max_pool( conv4_3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')

        conv5_1 = tf.nn.relu( self.conv2d(pool4,   [3,3,512], 'conv5_1', reuse) )
        conv5_2 = tf.nn.relu( self.conv2d(conv5_1, [3,3,512], 'conv5_2', reuse) )
        conv5_3 = tf.nn.relu( self.conv2d(conv5_2, [3,3,512], 'conv5_3', reuse) )
        pool5   = tf.nn.max_pool( conv5_3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
        pool5   = tf.contrib.layers.flatten( pool5 )
        print "pool5.shape", pool5.shape

        self.output = pool5

        return self.output

    def train_model(self):

        self.o1    = self.network(self.left, False)
        self.o2    = self.network(self.right, True)
        
        loss       = self.contrastive_loss( self.o1, self.o2, self.y)
        optimizer  = tf.train.AdamOptimizer( self.learning_rate)
        gradvars   = optimizer.compute_gradients( loss )
        #print [var for grad, var in gradvars if grad == None]
        capped     = [( tf.clip_by_value( grad, -1., 1.), var) for grad, var in gradvars ]        
        train_op   = optimizer.apply_gradients( capped )

        return loss, train_op

    def contrastive_loss(self, y_true, y_predict, y):
        """
        L = 1/(2N)sum_{i}( y*d^2 + (1-y)*max(margin -d, 0).^2 )
        where y = 1, match 
              y = 0, not match
        """
        d           = tf.reduce_sum( tf.pow( y_true - y_predict, 2), axis=1, keep_dims=True)
        d           = tf.sqrt(d)
        #print d.get_shape().as_list()
        tmp1        = y * tf.square(d)
        tmp2        = (1- y) * tf.square( tf.maximum((self.margin -d ), 0 ))
        #print tmp1.get_shape().as_list()
        #print tmp2.get_shape().as_list()
        out         = tf.reduce_mean(tmp1 + tmp2)/2
        #print out.get_shape().as_list()
        #exit(0)
        return out


if __name__ == '__main__':
    margin               = 0.2
    LR                   = 1e-3
    vgg_16_model_path    ='/home/fensi/nas/vgg16/vgg16.npy' 
    SiameseNet           = Siamese(vgg_16_model_path, margin=margin, learning_rate=LR )

    data_hnd             = data_handler.DataHandler() 
    data                 = data_hnd.get_mnist()
    train                = data.train.images
    label                = data.train.labels

    #SiameseNet.left
    #SiameseNet.train_model()