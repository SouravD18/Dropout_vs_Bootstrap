import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow_probability as tfp

from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils as np_utils

dense_depth=20
dense_units=20
dense_activation=tf.nn.relu
dropout_rate=0.5
output_units=1
output_activation=tf.nn.sigmoid
model_optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001)
model_loss=tf.keras.losses.binary_crossentropy
model_metrics=["acc", tf.keras.metrics.AUC()]

def add_dense_dropout(layer_list,
                      dense_depth,
                      dense_units,
                      dense_activation,
                      dropout_rate):
  for i in range(dense_depth-1):
    layer_list.append(tfp.layers.DenseFlipout(dense_units, dense_activation))

  layer_list.append(tfp.layers.DenseFlipout(dense_units,
                               dense_activation))

  return layer_list

## AVG_last_acc 0.9135958
## AVG_last_loss 0.36919246179171095

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epochs, logs={}):
    val_acc = logs.get("val_acc")
    val_loss = logs.get("val_loss")
    if val_acc > 0.9136 and val_loss < 0.3692:
      print("Stopping training as converging in accuracy and loss")
      self.model.stop_training = True
      return

def get_compiled_model(wtf=0):

  layer_list = []
  layer_list.append(tfkl.Flatten())
  layer_list = add_dense_dropout(layer_list,
                                dense_depth,
                                dense_units,
                                dense_activation,
                                dropout_rate)
  layer_list.append(tfp.layers.DenseFlipout(output_units,
                              output_activation))

  model = tf.keras.models.Sequential(layer_list)
  model.compile(optimizer=model_optimizer,
                loss=model_loss,
                metrics=model_metrics)
  
  return model

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set numeric type to float32 from uint8
x_train = x_train.astype("float16")
x_test = x_test.astype("float16")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform it into a binary classification problem: 3 or not
y_train[y_train != 3] = 0
y_train[y_train == 3] = 1

y_test[y_test != 3] = 0
y_test[y_test == 3] = 1

def unit_process():
  model = get_compiled_model()
  train_steps = len(x_train) // 128
  test_steps = len(x_test) // 128

  history = model.fit(x=x_train[:train_steps * 128],
                      y=y_train[:train_steps * 128],
                      epochs=100,
                      batch_size=128,
                      validation_data= 
                      (x_test[:test_steps * 128], y_test[:test_steps * 128]),
                      verbose=0,
                      callbacks=[myCallback()])
  return model, history

model, history = unit_process()
print("Done Training")
n=1000
test_steps = len(x_test) // 128
result = []
for i in range(1000):
    prediction = model.predict(x_test[:test_steps * 128])
    result.append(prediction)

all_predictions = np.array(result)

fname = "dense_flipout_results"
pickle.dump(all_predictions, open(fname, "wb"))
print("Done Everything")
