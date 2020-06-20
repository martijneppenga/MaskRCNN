# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:18:36 2020

@author: meppenga
"""
import tensorflow.keras as keras
import tensorflow as tf

loss_tracker = keras.metrics.Mean(name="loss")   
class CustomModel(keras.Model):
    
    def addConfigfile(self,config):
        self.config = config
    
    def train_step(self, data):
        # Unpack data
        Inputs, Outputs = data

        with tf.GradientTape() as tape:
            # y_pred =  # Forward pass
            _, _, _, _, _, _, _, _, _, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = self(Inputs, training=True) 
            
            
            loss = [tf.reduce_mean(rpn_class_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("rpn_class_loss", 1.),
                    tf.reduce_mean(rpn_bbox_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("rpn_bbox_loss", 1.), 
                    tf.reduce_mean(class_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_class_loss", 1.)
                    , tf.reduce_mean(bbox_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_bbox_loss", 1.), 
                    tf.reduce_mean(mask_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_mask_loss", 1.)]


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        # mae_metric.update_state(Inputs, y_pred)
        return {"loss": loss_tracker.result()}
    

    def test_step(self, data):
        # Unpack the data
        Inputs, Outputs = data
        # Compute predictions
        _, _, _, _, _, _, _, _, _, rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss = self(Inputs, training=True)
        loss = [tf.reduce_mean(rpn_class_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("rpn_class_loss", 1.),
                    tf.reduce_mean(rpn_bbox_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("rpn_bbox_loss", 1.), 
                    tf.reduce_mean(class_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_class_loss", 1.)
                    , tf.reduce_mean(bbox_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_bbox_loss", 1.), 
                    tf.reduce_mean(mask_loss, keepdims=True) * self.config.LOSS_WEIGHTS.get("mrcnn_mask_loss", 1.)] # Updates the metrics tracking the loss
        loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": loss_tracker.result()}
