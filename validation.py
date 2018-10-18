from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import  pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("./california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = california_housing_dataframe[
        ['latitude',
        'longitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']]

    processed_features = selected_features.copy()

    # Create a syntetic feature
    processed_features['rooms_per_person'] = (
        california_housing_dataframe['total_rooms'] /
        california_housing_dataframe['population'])

    return processed_features

def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
            from the California housing data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the targets to be in units of thousands of dollars.
    output_targets['median_house_value'] = (
        california_housing_dataframe['median_house_value'] / 1000.0)

    return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
# print(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# print(training_targets.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
# print(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# print(validation_targets.describe())


def plot_lat_lon_vs_median_house_value(validation_examples, validation_targets, training_examples, training_targets):
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validation Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"],
                validation_examples["latitude"],
                cmap="coolwarm",
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

    ax = plt.subplot(1,2,2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    _ = plt.plot()
    plt.show()

# plot_lat_lon_vs_median_house_value(validation_examples, validation_targets, training_examples, training_targets)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    # Convert pandas data into dict of np arrays.
    features = {key:np.array(value) for key, value in dict(features).items()}

    # Construct a dataset and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
        input_features: The names of the numerical input features to use.
    Returns:
        A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
        for my_feature in input_features])


def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a linear regression model of multiple features.

        In addition to training, this function also prints training progress information,
        as well as a plot of the training and validation loss over time.

    Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns from
            `california_housing_dataframe` to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column from
            `california_housing_dataframe` to use as target for validation.

    Returns:
        A `LinearRegressor` object trained on the training data.
    """
    periods = 10
    steps_per_period = steps/periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 1. Create input functions.
    training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(
        training_examples,
        training_targets['median_house_value'],
        shuffle=False,
        num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(
        validation_examples,
        validation_targets['median_house_value'],
        shuffle=False,
        num_epochs=1)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # 2. Take a break and compute predictions.
        training_predictions = linear_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # Occasionally print the current loss.
        print("\t period %02d : %0.2f" % (period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor


linear_regressor = train_model(
    learning_rate=0.00003,
    steps=300,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

households_weight = linear_regressor.get_variable_value('linear/linear_model/households/weights')[0]
housing_median_age_weight = linear_regressor.get_variable_value('linear/linear_model/housing_median_age/weights')[0]
latitude_weight = linear_regressor.get_variable_value('linear/linear_model/latitude/weights')[0]
longitude_weight = linear_regressor.get_variable_value('linear/linear_model/longitude/weights')[0]
median_income_weight = linear_regressor.get_variable_value('linear/linear_model/median_income/weights')[0]
population_weight = linear_regressor.get_variable_value('linear/linear_model/population/weights')[0]
rooms_per_person_weight = linear_regressor.get_variable_value('linear/linear_model/rooms_per_person/weights')[0]
total_bedrooms_weight = linear_regressor.get_variable_value('linear/linear_model/total_bedrooms/weights')[0]
total_rooms_weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')[0]


used_features = [
    'latitude',
    'longitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'rooms_per_person']


# predicted_weights = [
#     linear_regressor.get_variable_value('linear/linear_model/%s/weights' % f)[0]
#     for f in used_features] # 1x9

predicted_weights = pd.DataFrame()
for f in used_features:
    predicted_weights[f] = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % f)[0]

predicted_bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

sample = california_housing_dataframe.sample(300)
sample_features = preprocess_features(sample) # 300x9
sample_targets = sample['median_house_value'].apply(lambda val: val / 1000.0) # 300x1

sample_features_T = sample_features.T # 9x300


predictions = predicted_bias + predicted_weights.dot(sample_features_T) # 1x9 . 9x300 = 1x300

import pdb; pdb.set_trace()


cmp = bias + (sample[0:1]['longitude'] * longitude_weight ) + (sample[0:1]['latitude'] * latitude_weight) + (sample[0:1]['housing_median_age'] * housing_median_age_weight)+ (sample[0:1]['total_rooms'] * total_rooms_weight) + (sample[0:1]['total_bedrooms'] * total_bedrooms_weight) + (sample[0:1]['population'] * population_weight) + (sample[0:1]['households'] * households_weight) + (sample[0:1]['median_income'] * median_income_weight)


root_mean_squared_error = math.sqrt(
      metrics.mean_squared_error(predictions, sample_targets))

print(predictions)

# sample_features = [sample_features[f] for f in used_features] # 300x9



# plt.figure(figsize=(15, 8))
# plt.subplot(1, 2, 1)

# # plt.scatter(linear_regressor["predictions"], linear_regressor["targets"])
# plt.show()









# (Pdb) sample[0:2]
#       Unnamed: 0  longitude  latitude  housing_median_age  total_rooms  total_bedrooms  population  households  median_income  median_house_value
#16973       16973     -124.2      40.8                39.0       1606.0           330.0       731.0       327.0            1.6             68300.0
#14391       14391     -122.1      37.9                22.0       4949.0           626.0      1850.0       590.0           10.5            500001.0
#(Pdb)

# (Pdb) pp predictions[0:2]
#    16973  14391
# 0   73.1  212.2
