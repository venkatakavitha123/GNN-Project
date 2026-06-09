import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pickle

dataset = pd.read_csv("Dataset/Dataset.csv")

label_encoder = []
columns = dataset.columns
types = dataset.dtypes.values
for j in range(len(types)):
    name = types[j]
    if name == 'object': #finding column with object type
        le = LabelEncoder()
        dataset[columns[j]] = pd.Series(le.fit_transform(dataset[columns[j]].astype(str)))#encode all str columns to numeric
        label_encoder.append([columns[j], le])
dataset.fillna(0, inplace = True)
Y = dataset['Status'].ravel()
Y = to_categorical(Y)
dataset.drop(['Location', 'Status'], axis = 1,inplace=True)

X = dataset.values

scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#training graphCNN algorithm
#Create GNN model to detect fault from all services
graph_conv_filters = np.eye(1)
graph_conv_filters = K.constant(graph_conv_filters)
graph_model = Sequential()
graph_model.add(GraphCNN(16, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
graph_model.add(GraphCNN(8, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
graph_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(0.001)))
graph_model.add(Dense(units = 32, activation = 'elu'))
graph_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
graph_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/1gnn_weights.h5") == False:
    hist = graph_model.fit(X, Y, batch_size=1, epochs=60, validation_data = (X_test, y_test), verbose=1)
    graph_model.save_weights("model/gnn_weights.h5")
    f = open("model/gnn_history.pckl", 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    graph_model.load_weights("model/gnn_weights.h5")
#perform prediction on test data of all services and calculate accuracy and other metrics
pred = []
for i in range(len(X_test)):
    temp = []
    temp.append(X_test[i])
    temp = np.asarray(temp)
    predict = graph_model.predict(temp, batch_size=1)
    predict = np.argmax(predict)
    pred.append(predict)
y_tested = np.argmax(y_test, axis=1)    
predict = np.asarray(pred)
acc = accuracy_score(y_tested, predict)
print(acc)























