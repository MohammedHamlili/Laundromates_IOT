import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

MODEL_FILE = 'nn_model.hd5'
DATA_FILE = 'data.csv'

def process_data(data_name):
    #open csv file
    with open(data_name, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    #convert columns into strings (bools for results column)
    for i in range(1,len(data)):
        for j in range(len(data[i])-1):
            data[i][j] = float(data[i][j])
        data[i][-1] = bool(int(data[i][-1]))

    #organising in dataframes
    pre_data = pd.DataFrame(data)
    x_data = pre_data.iloc[1:,:-1].dropna()
    y_data = pre_data.iloc[1:,-1].dropna()

    #scaling data to [0,1]
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    X = pd.DataFrame(scaler.transform(x_data))

    #adding noise to our data to increase data size
    noisy_X = pd.DataFrame()
    mu=0
    for i in range(X.shape[1]):
        #standard deviation of each column that corresponds to 'off' state 
        std = X[(~y_data.astype(bool)).values][:][i].std()
        print(std)
        #std = X[y_data.values][:][i].std()
        noise = np.random.normal(mu, std, size = X[:][i].shape)
        noisy_X[i] = X[:][i] + noise

    X = pd.concat([X, noisy_X], axis=0)
    Y = pd.DataFrame(to_categorical(y = y_data, num_classes = 2))     
    Y = pd.concat([Y, Y], axis=0)    
    
    #split training from test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
    
    return (x_data, X_train, X_test, Y_train, Y_test)

def build_train_model(model_name, x_data, X_train, X_test, Y_train, Y_test):
    #BUILD
    # Create the neural network
    nn = Sequential()
    nn.add(Dense(100, input_shape = (4, ), activation = 'relu'))
    nn.add(Dense(2, activation = 'softmax'))

    # Create our optimizer
    sgd = SGD(learning_rate = 0.2)

    #TRAIN
    # 'Compile' the network to associate it with a loss function,
    # an optimizer, and what metrics we want to track
    nn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = 'accuracy')
    
    #save file as '.hd5'
    checkpoint = ModelCheckpoint(model_name)
    
    nn.fit(X_train, Y_train, shuffle = True, epochs = 100, validation_data = (X_test, Y_test), callbacks=[checkpoint])
    
    #save max and min values for later scaling
    update_extremes(x_data, model_name)

def update_extremes(x_values, model_name):
    f = open(model_name + "/extremes.csv","w")
    for i in range(x_values.shape[1]):
        max_val = x_values.iloc[:,i].max()
        min_val = x_values.iloc[:,i].min()
        f.write(str(min_val) + "; " + str(max_val) + "\n")
    
def main():
   (x_data, X_train, X_test, Y_train, Y_test) = process_data(DATA_FILE)
   #parameters to test out: epochs, learning_rate, amount of layers, width of layers, loss, activation
   build_train_model(MODEL_FILE, x_data, X_train, X_test, Y_train, Y_test)

if __name__ == '__main__':
    main()
