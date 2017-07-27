import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from utils import ProgressBar, plot


class VanillaInfoGAN(object):

    def __init__(self, config, sess):
        self.input_dim = config.input_dim
        self.z_dim = config.z_dim
        self.c_cat = config.c_cat
        self.c_cont = config.c_cont
        self.d_update = config.d_update
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.lr = config.lr
        self.max_grad_norm = config.max_grad_norm
        self.use_adam = config.use_adam
        self.show_progress = config.show_progress

        if self.use_adam:
            self.optimizer = tf.train.AdamOptimizer
        else:
            self.optimizer = tf.train.GradientDescentOptimizer

        self.checkpoint_dir = config.checkpoint_dir
        self.image_dir = config.image_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        self.random_seed = 42

        self.X = tf.placeholder(tf.float32, [None, self.input_dim], 'X')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        self.c_i = tf.placeholder(tf.float32, [None, self.c_cat], 'c_cat')
        self.c_j = tf.placeholder(tf.float32, [None, self.c_cont], 'c_cont')
        self.c = tf.concat([self.c_i, self.c_j], axis=1)
        self.z_c = tf.concat([self.z, self.c_i, self.c_j], axis=1)

        self.sess = sess


    def z_sampler(self, dim1):
        return np.random.normal(-1, 1, size=[dim1, self.z_dim])


    def c_cat_sampler(self, dim1):
        return np.random.multinomial(1, [0.1] * self.c_cat, size=dim1)


    def c_cont_sampler(self, dim1):
        return np.random.uniform(-1, 1, size=[dim1, self.c_cont])


    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

        return tf.random_normal(shape=size, stddev=xavier_stddev)

    
    def build_generator(self):
        self.G_W1 = tf.Variable(self.xavier_init([self.z_dim + self.c_cat + self.c_cont, 128]))
        self.G_b1 = tf.Variable(tf.zeros([128]))
        self.G_W2 = tf.Variable(self.xavier_init([128, self.input_dim]))
        self.G_b2 = tf.Variable(tf.zeros([self.input_dim]))

        G_layer1 = tf.nn.relu(tf.matmul(self.z_c, self.G_W1) + self.G_b1)
        G_layer2 = tf.nn.sigmoid(tf.matmul(G_layer1, self.G_W2) + self.G_b2)

        self.G = G_layer2


    def build_discriminator(self):
        self.D_W1 = tf.Variable(self.xavier_init([self.input_dim, 128]))
        self.D_b1 = tf.Variable(tf.zeros([128]))
        self.D_W2 = tf.Variable(self.xavier_init([128, 1]))
        self.D_b2 = tf.Variable(tf.zeros([1]))

        D_real_layer1 = tf.nn.relu(tf.matmul(self.X, self.D_W1) + self.D_b1)
        D_real_layer2 = tf.nn.sigmoid(tf.matmul(D_real_layer1, self.D_W2) + self.D_b2)

        D_fake_layer1 = tf.nn.relu(tf.matmul(self.G, self.D_W1) + self.D_b1)
        D_fake_layer2 = tf.nn.sigmoid(tf.matmul(D_fake_layer1, self.D_W2) + self.D_b2)

        self.D_real = D_real_layer2
        self.D_fake = D_fake_layer2


    def build_Q(self):
        self.Q_W1 = tf.Variable(self.xavier_init([self.input_dim, 128]))
        self.Q_b1 = tf.Variable(tf.zeros([128]))
        self.Q_W2 = tf.Variable(self.xavier_init([128, self.c_cat + self.c_cont]))
        self.Q_b2 = tf.Variable(tf.zeros([self.c_cat + self.c_cont]))

        Q_layer1 = tf.nn.relu(tf.matmul(self.G, self.Q_W1) + self.Q_b1)
        Q_layer2 = tf.matmul(Q_layer1, self.Q_W2) + self.Q_b2
        Q_layer2_cat = tf.nn.softmax(Q_layer2[:, :self.c_cat])
        Q_layer2_cont = tf.nn.sigmoid(Q_layer2[:, self.c_cat:])

        Q_c_given_x = tf.concat([Q_layer2_cat, Q_layer2_cont], axis=1)

        self.Q_c_given_x = Q_c_given_x


    def build_model(self):
        self.build_generator()
        self.build_discriminator()
        self.build_Q()

        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake + 1e-8))
        self.D_loss = -tf.reduce_mean(tf.log(self.D_real + 1e-8) + tf.log(1 - self.D_fake + 1e-8))

        cond_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_c_given_x + 1e-8) * self.c, axis=1))
        ent = 1
        self.Q_loss = cond_ent + ent
        
        G_params = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]
        D_params = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
        Q_params = [self.Q_W1, self.Q_W2, self.Q_b1, self.Q_b2]

        self.var_list = {'G_W1': self.G_W1, 'G_W2': self.G_W2, 'G_b1': self.G_b1, 'G_b2': self.G_b2, \
                         'D_W1': self.D_W1, 'D_W2': self.D_W2, 'D_b1': self.D_b1, 'D_b2': self.D_b2, \
                         'Q_W1': self.Q_W1, 'Q_W2': self.Q_W2, 'Q_b1': self.Q_b1, 'Q_b2': self.Q_b2}

        G_optimizer = self.optimizer(self.lr)
        G_grads_and_vars = G_optimizer.compute_gradients(self.G_loss, G_params)
        G_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in G_grads_and_vars]
        self.G_optim = G_optimizer.apply_gradients(G_clipped_grads_and_vars)

        D_optimizer = self.optimizer(self.lr)
        D_grads_and_vars = D_optimizer.compute_gradients(self.D_loss, D_params)
        D_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in D_grads_and_vars]
        self.D_optim = D_optimizer.apply_gradients(D_clipped_grads_and_vars)

        Q_optimizer = self.optimizer(self.lr)
        Q_grads_and_vars = Q_optimizer.compute_gradients(self.Q_loss, G_params + Q_params)
        Q_clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in Q_grads_and_vars]
        self.Q_optim = Q_optimizer.apply_gradients(Q_clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(var_list=self.var_list)


    def train(self):
        avg_G_loss = 0
        avg_D_loss = 0
        avg_Q_loss = 0
        iterations = int(self.mnist.train.num_examples / self.batch_size)

        if self.show_progress:
            bar = ProgressBar('Train', max=iterations)

        for i in range(iterations):

            if self.show_progress:
                bar.next()

            batch_xs, _ = self.mnist.train.next_batch(self.batch_size)
            feed_dict = {self.X: batch_xs, \
                         self.z: self.z_sampler(self.batch_size), \
                         self.c_i: self.c_cat_sampler(self.batch_size), \
                         self.c_j: self.c_cont_sampler(self.batch_size)}

            for _ in range(self.d_update):
                _, D_loss = self.sess.run([self.D_optim, self.D_loss], feed_dict=feed_dict)
            _, G_loss = self.sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)
            _, Q_loss = self.sess.run([self.Q_optim, self.Q_loss], feed_dict=feed_dict)
            
            avg_G_loss += G_loss / iterations
            avg_D_loss += D_loss / iterations
            avg_Q_loss += Q_loss / iterations

        if self.show_progress:
            bar.finish()

        return avg_G_loss, avg_D_loss, avg_Q_loss


    def run(self):

        for epoch in range(self.nepoch):
            avg_G_loss, avg_D_loss, avg_Q_loss = self.train()

            state = {'G Loss': '{:.5f}'.format(avg_G_loss), \
                     'D Loss': '{:.5f}'.format(avg_D_loss), \
                     'Q Loss': '{:.5f}'.format(avg_Q_loss), \
                     'Epoch': epoch}

            print(state)

            if epoch % 5 == 0:
                feed_dict = {self.z: self.z_sampler(16), self.c_i: self.c_cat_sampler(16), self.c_j: self.c_cont_sampler(16)}
                samples = self.sess.run(self.G, feed_dict=feed_dict)
                fig = plot(samples)
                plt.savefig(os.path.join(self.image_dir, '{:04d}.png'.format(epoch)), bbox_inches='tight')
                plt.close(fig)

                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'InfoGAN.model'))


    def generate(self, c_cat, c_cont):
        self.load()

        return self.sess.run(self.G, feed_dict={self.z: self.z_sampler(len(c_cat)), self.c_i: c_cat, self.c_j: c_cont})


    def extract_features(self, X):
        self.load()

        return self.sess.run(self.Q_c_given_x, feed_dict={self.G: X})


    def load(self):
        print('[*] Reading Checkpoints...')
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('[!] No Checkpoint Found')

