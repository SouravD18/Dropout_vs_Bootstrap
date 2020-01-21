import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras.layers as tfkl

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
  for i in range(dense_depth):
    layer_list.append(tfkl.Dense(dense_units,
                                 dense_activation))
    layer_list.append(tfkl.Dropout(dropout_rate))

  return layer_list

def get_compiled_model(wtf=0):

  layer_list = []
  layer_list.append(tfkl.Flatten())
  layer_list = add_dense_dropout(layer_list,
                                dense_depth,
                                dense_units,
                                dense_activation,
                                dropout_rate)
  layer_list.append(tfkl.Dense(output_units,
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

  sample_size = train_steps * 128
  sample_idx = np.random.choice(
      np.arange(sample_size), sample_size)


  history = model.fit(x=x_train[sample_idx],
                      y=y_train[sample_idx],
                      epochs=100,
                      batch_size=128,
                      validation_data= 
                      (x_test[:test_steps * 128], y_test[:test_steps * 128]),
                      verbose=0)
  result = model.predict(x_test[:test_steps * 128])
  return result, history

predictions, history = unit_process()

fname = str(np.random.randint(10**7))

predictions.dump("./predictions/"+fname)

hfile = open("./history/"+fname, "wb")
pickle.dump(history.history, hfile)
hfile.close()

