# Data Mining Project
# Shahla Atapoor
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# Data Mining course's slides
import pandas
import shutil
import numpy
import os
import matplotlib.pyplot as plt
# Keras library requirements(From lectures)
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
# sklearn library requirements
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# load the data
df = pandas.read_csv('C:/Users/HP/PycharmProjects/untitled/accel_x_final_dataset.csv')
dataset = df.values # get the data frame as python list
# make sure everything's in float
X_axis = dataset[:,0:12].astype(float)
Y_axis = dataset[:,12].astype(int)

# Function to create model, required for KerasClassifier
def model_creating():
    """
    Keras Fully Connected Neural Network
    :return: Keras Model
    """
    global model
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=12, activation='relu')) # hidden 1
    model.add(Dense(15, activation='relu')) # hidden 2
    model.add(Dense(15, activation='relu')) # hidden 3
    model.add(Dense(2, activation='softmax')) # output layer
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class LossHistory(Callback):# Create a callback: Callback that records events into a History object
    """
    store the calculated loss from the network
    """
    def on_train_begin(self, logs={}):
        """

        :param logs:
        :return:
        """
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        """

        :param batch:
        :param logs:
        :return:
        """
        self.losses.append(logs.get('loss'))


history = LossHistory() # https://keras.io/callbacks/#example-recording-loss-history
stopping = EarlyStopping(monitor='val_acc', patience=20) # https://keras.io/callbacks/#earlystopping

# create the model
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
estimator =  (model_creating, epochs=200, batch_size=100, verbose=2)

# split the data into test and train with 0.33 rate
X_train_data, X_test_data, Y_train_data, Y_test_data = train_test_split(X_axis, Y_axis, test_size=0.33, random_state=5)
Y_test_data = to_categorical(Y_test_data) # Model checkpoint

# fit the estimator / train the model
train_results = estimator.fit(X_train_data, Y_train_data, callbacks=[history, stopping],
                        validation_data=(X_test_data, Y_test_data))

# 10-fold cross validation
k_fold_cross_validation = KFold(n_splits=10, shuffle=True, random_state=5)
# result of cross-validation
cross_validation_results = cross_val_score(estimator, X_test_data, Y_test_data, cv=k_fold_cross_validation)

# print and plot the results
print("The result of network on test data: %.2f%% (%.2f%%)" % (cross_validation_results.mean()*100,
                                                               cross_validation_results.std()*100))
figsize = (15, 5)
fig, ax = plt.subplots(figsize=figsize)
ax.plot(train_results.history['acc'], linewidth=2, color="black")
plt.figure()
plt.show()
