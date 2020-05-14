import tensorflow as tf
import numpy as np 
import cv2
import pickle
import glob 
import os 
import time 
import argparse
import ipdb

from network import network 
from loss import total_loss
from dataloader import dataloader
class BlazeFace():
    
    def __init__(self, config):
        
        self.input_shape = (config.input_shape, config.input_shape, 3)
        self.feature_extractor = network(self.input_shape)
        
        self.n_boxes = [1, 1, 1, 1] # 2 for each of 16x16, 8x8, 4x4, 2x2
        self.num_classes = 11 # class 0 = background class, classes 1-10 for each of the MNIST digit

        self.model = self.build_model()
        
        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch
            
        self.checkpoint_path = config.checkpoint_path
        self.numdata = config.numdata
        
    def build_model(self):
        
        model = self.feature_extractor
        
        # 16x16 bounding box - Confidence, [batch_size, 16, 16, 2*11]
        bb_16_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * self.num_classes, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[0])
        # reshape [batch_size, 16**2 * #bbox(2), num_classes]
        bb_16_conf_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], self.num_classes))(bb_16_conf)
        
        
        # 8 x 8 bounding box - Confidence, [batch_size, 8, 8, 2*11]
        bb_8_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * self.num_classes, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[1])
        # reshape [batch_size, 8**2 * #bbox(2), num_classes]
        bb_8_conf_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], self.num_classes))(bb_8_conf)

        
        # 4 x 4 bounding box - Confidence, [batch_size, 4, 4, 2*11]
        bb_4_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[2] * self.num_classes, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[2])
        # reshape [batch_size, 4**2 * #bbox(2), num_classes]
        bb_4_conf_reshaped = tf.keras.layers.Reshape((4**2 * self.n_boxes[2], self.num_classes))(bb_4_conf)

        
        # 2 x 2 bounding box - Confidence, [batch_size, 2, 2, 2*11]
        bb_2_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[3] * self.num_classes, 
                                            kernel_size=3, 
                                            padding='same', 
                                            activation='sigmoid')(model.output[3])
        # reshape [batch_size, 2**2 * #bbox(2), num_classes]
        bb_2_conf_reshaped = tf.keras.layers.Reshape((2**2 * self.n_boxes[3], self.num_classes))(bb_2_conf)
        
        # Concatenate confidence prediction 
        # shape : [batch_size, 680, 11] , 680 = (16**2 + 8**2 + 4**2 + 2**2) * 2
        conf_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped, bb_4_conf_reshaped, bb_2_conf_reshaped])
        
        
        # 16x16 bounding box - loc [x, y, w, h]
        bb_16_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 4,
                                            kernel_size=3, 
                                            padding='same')(model.output[0])
        # [batch_size, 16**2 * #bbox(2), 4]
        bb_16_loc_reshaped = tf.keras.layers.Reshape((16**2 * self.n_boxes[0], 4))(bb_16_loc)
        
        
        # 8x8 bounding box - loc [x, y, w, h]
        bb_8_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 4,
                                          kernel_size=3,
                                          padding='same')(model.output[1])
        bb_8_loc_reshaped = tf.keras.layers.Reshape((8**2 * self.n_boxes[1], 4))(bb_8_loc)
        
        # 4x4 bounding box - loc [x, y, w, h]
        bb_4_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[2] * 4,
                                          kernel_size=3,
                                          padding='same')(model.output[2])
        bb_4_loc_reshaped = tf.keras.layers.Reshape((4**2 * self.n_boxes[2], 4))(bb_4_loc)
        
        # 2x2 bounding box - loc [x, y, w, h]
        bb_2_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[3] * 4,
                                          kernel_size=3,
                                          padding='same')(model.output[3])
        bb_2_loc_reshaped = tf.keras.layers.Reshape((2**2 * self.n_boxes[3], 4))(bb_2_loc)
        
        
        # Concatenate  location prediction 
        # shape : [batch_size, 680, 4] , 680 = (16**2 + 8**2 + 4**2 + 2**2) * 2
        loc_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped, bb_4_loc_reshaped, bb_2_loc_reshaped])
        
        # shape : [batch_size, 680, 11+4] , 680 = (16**2 + 8**2 + 4**2 + 2**2) * 2
        output_combined = tf.keras.layers.Concatenate(axis=-1)([conf_of_bb, loc_of_bb])
        
        # Detectors model 
        return tf.keras.models.Model(model.input, output_combined)
    

    def train(self):
        
        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model = self.model
        model.compile(loss=total_loss, optimizer=opt)

        """ Callback """
        monitor = 'loss'
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=4)

        """Callback for Tensorboard"""
        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """

        ## Total Dataset = 2625
        STEP_SIZE_TRAIN = self.numdata // self.batch_size

        t0 = time.time()
        
        data_gen = dataloader(config.dataset_dir, config.label_path, self.num_classes, self.batch_size)

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=data_gen,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            
            print(res.history)
            
            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % 100 == 0:
                model.save_weights(os.path.join(config.checkpoint_path, str(epoch)))

        print('Total training time : %.1f' % (time.time() - t0))



if __name__ == "__main__":

    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=1) #128
    args.add_argument('--nb_epoch', type=int, default=1) #1000
    args.add_argument('--numdata', type=int, default=1) 
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    args.add_argument('--dataset_dir', type=str, default="data/img")
    args.add_argument('--label_path', type=str, default="data/labels.pkl")

    config = args.parse_args()

    blazeface = BlazeFace(config)

    if config.train:
        blazeface.train()


