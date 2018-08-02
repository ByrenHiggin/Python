# Licenced under the Creative Commons Attribution-ShareAlike 4.0 licence.
# https://creativecommons.org/licenses/by-sa/4.0/ 
#
# Feel free to use this all you want, regardless of changes made. Just make sure you give appropriate credit
# and indicate any changes made (feel free to fork and pull this with your own changes so I can see how it could be improved).
# I would like this to be used non-commercially, but if you have a cool application idea for this let me know and maybe we 
# can make something together. Cheers, Byren Higgin (https://github.com/ByrenHiggin)

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm, gridspec, pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

DebugMode = False

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

california_housing_test_data = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_test.csv", sep=",")

class HyperParameters:
  def __init__(self,steps=1000,learning_rate=0.1,batch_size=200,periods=10,hidden_units=[10,10],clip_range=5.0,L1=0.001,L2=0.001):
    self.steps = steps
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.periods = periods
    self.hidden_units = hidden_units
    self.clip_range = clip_range
    self.L1 = L1
    self.L2 = L2

def preprocess_features(dataframe,feature_set,MultiplyCrossFeatures=None,DivideCrossFeatures=None):
  selectedFeatures = dataframe[feature_set]
  processedFeatures = selectedFeatures.copy()
  if MultiplyCrossFeatures is not None:
    for key in DivideCrossFeatures.items():
      m = None
      for items in key[1]:
        if m is None:
          m = selectedFeatures[items]
        else:
          m = m * selectedFeatures[items]
      processedFeatures[key[0]] = m
  if DivideCrossFeatures is not None:
    for key in DivideCrossFeatures.items():
      l = None
      for items in key[1]:
        if l is None:
          l = selectedFeatures[items]
        else:
          l = l / selectedFeatures[items]
      processedFeatures[key[0]] = l
  return processedFeatures

def preprocess_targets(dataframe,feature_set, ModFunction = None):
  output_targets = pd.DataFrame()
  output_targets[feature_set] = dataframe[feature_set]
  if ModFunction is not None:
    output_targets[feature_set] = output_targets[feature_set].apply(ModFunction)
  return output_targets

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
  features = {key:np.array(value) for key,value in dict(features).items()}
  ds = Dataset.from_tensor_slices((features,targets))
  ds = ds.batch(batch_size).repeat(num_epochs)
  if shuffle:
    ds = ds.shuffle(10000)
    
  features, labels = ds.make_one_shot_iterator().get_next()
  return features, labels

def test_nn_regression_model(
  model,
  test_examples,
  test_targets):
  
  test_input_fn = lambda: my_input_fn(test_examples,test_targets, num_epochs=1, shuffle=False)
  
  testing_predictions = model.predict(input_fn=test_input_fn)
  testing_predictions = np.array([item['predictions'][0] for item in testing_predictions])
  testing_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(testing_predictions,test_targets))
  print("Final RMSE (on test data): %0.2f" % testing_root_mean_squared_error)
  return  testing_root_mean_squared_error
  
def train_nn_regression_model(
  hp,
  training_examples,
  training_targets,
  validation_examples,
  validation_targets):
    
  periods = hp.periods
  steps = hp.steps
  steps_per_period = steps / periods
  
  optimizer = tf.train.FtrlOptimizer(learning_rate=hp.learning_rate,l1_regularization_strength=hp.L1,l2_regularization_strength=hp.L2)
  optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer,hp.clip_range)
  
  dnn_regressor = tf.estimator.DNNRegressor(
    feature_columns=construct_feature_columns(training_examples),
      hidden_units=hp.hidden_units,
      optimizer=optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples,training_targets, batch_size = hp.batch_size)
  predicti_input_fn = lambda: my_input_fn(training_examples,training_targets, num_epochs=1, shuffle=False)
  validiti_input_fn = lambda: my_input_fn(validation_examples,validation_targets, num_epochs=1, shuffle=False)
  
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    training_predictions = dnn_regressor.predict(input_fn=predicti_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions]) 
    
    validation_predictions = dnn_regressor.predict(input_fn=validiti_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions,training_targets))
    validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions,validation_targets))
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor
 
  
DivideByThousand = lambda x: x / 1000

hp = HyperParameters(learning_rate=0.001,steps=2000,batch_size=100,hidden_units=[10,5])

training_examples = preprocess_features(dataframe=california_housing_dataframe.head(12000),
                    feature_set=["latitude","longitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"],
                    DivideCrossFeatures={"rooms_per_person":["total_rooms","population"]})
training_targets = preprocess_targets(dataframe=california_housing_dataframe.head(12000),feature_set=["median_house_value"],ModFunction=DivideByThousand)

validation_examples = preprocess_features(dataframe=california_housing_dataframe.tail(5000),
                    feature_set=["latitude","longitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"],
                    DivideCrossFeatures={"rooms_per_person":["total_rooms","population"]}
                   )
validation_targets = preprocess_targets(dataframe=california_housing_dataframe.tail(5000),feature_set=["median_house_value"],ModFunction=DivideByThousand)

test_examples = preprocess_features(dataframe=california_housing_test_data,
                            feature_set=["latitude","longitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"],
                            DivideCrossFeatures={"rooms_per_person":["total_rooms","population"]})
test_targets = preprocess_targets(dataframe=california_housing_test_data,feature_set=["median_house_value"],ModFunction=DivideByThousand)

if DebugMode:
  
  print("Training examples summary:")
  display.display(training_examples.describe())
  print("Validation examples summary:")
  display.display(validation_examples.describe())
  print("Training targets summary:")
  display.display(training_targets.describe())
  print("Validation targets summary:")
  display.display(validation_targets.describe())
  print("Testing targets summary:")
  display.display(test_examples.describe())
  print("Testing targets summary:")
  display.display(test_targets.describe())

nn = train_nn_regression_model(hp,training_examples,training_targets,validation_examples,validation_targets)
test = test_nn_regression_model(nn,test_examples,test_targets)

