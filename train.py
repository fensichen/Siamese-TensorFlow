#train.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_handler
import model

# parameters
margin      = 0.2
NUM_STEP    = 1000
LR          = 1e-3
batch_size  = 30

data_hnd    = data_handler.DataHandler() 
data        = data_hnd.get_mnist()
train       = data.train.images
label       = data.train.labels
data_hnd.preprocess( train, label )
test_data   = np.array([im.reshape((28,28,1)) for im in data.test.images] )

vgg_16_model_path    ='/home/fensi/nas/vgg16/vgg16.npy' 
SiameseNet           = model.Siamese(vgg_16_model_path, margin=margin, learning_rate=LR )

train_loss, train_op = SiameseNet.train_model()

#with tf.name_scope("similarity"):
#   labels  = tf.placeholder(tf.int32, [None, 1], name='label')
#   labels  = tf.to_float(label)


#train_op    = tf.train.GradientDescentOptimizer(lr=LR).minimize(siamese.contrastive_loss())

conf        = tf.ConfigProto( gpu_options=tf.GPUOptions(allow_growth=True), device_count={'GPU':1}  ) 
#saver       = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range( NUM_STEP ):
        batch_left, batch_right, batch_sim = data_hnd.next_batch(batch_size)

        #print "batch_left", batch_left
        feed = { SiameseNet.left: batch_left, SiameseNet.right: batch_right, SiameseNet.y: batch_sim }
        
        tf_loss, tf_op = sess.run([ train_loss, train_op ], feed_dict=feed )
        print "loss: ", str(tf_loss).zfill(5)

    #   if (i + 1) % step == 0:
    #       feat = sess.run([])
            # plot result