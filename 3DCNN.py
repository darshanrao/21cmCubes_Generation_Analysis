import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import time
import h5py
from tensorflow.keras.callbacks import EarlyStopping


# get the start time
st = time.time()

# # Loading Data File
# hf = h5py.File('./Data/Data_bt.h5', 'r')
# hf.keys()
# temp_box=hf.get('dataset')


temp_box_np=np.load('/tier2/tifr/darshan/testrun/Noise_Data/signal_noise_sigma_10.npy')

#Loading CSV File
df= pd.read_csv('./Data/Data.csv')
print(df.head())

def normalized(arr):
    if np.std(arr) != 0:
        arr=(arr - np.mean(arr)) / np.std(arr)
    return arr

for i in range(0,10820):
    temp_box_np[i]=normalized(temp_box_np[i])


X=temp_box_np
y=np.array(df["Neutral Fraction"])

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.reshape(X_train.shape[0],128,128,128,1)
X_test = X_test.reshape(X_test.shape[0],128,128,128,1)



# Define input shape
input_shape = (128, 128, 128, 1)

# Define input layer
inputs = Input(shape=input_shape)

# Define convolutional layers
x = Conv3D(32, kernel_size=(3, 3, 3), activation=LeakyReLU(alpha=0.1))(inputs)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(64, kernel_size=(3, 3, 3), activation=LeakyReLU(alpha=0.1))(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(128, kernel_size=(3, 3, 3), activation=LeakyReLU(alpha=0.1))(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(256, kernel_size=(3, 3, 3), activation=LeakyReLU(alpha=0.1))(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)

# Define flatten layer
x = Flatten()(x)

# Define fully connected layers with dropout and batch normalization
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(1)(x)

# Define model
model = Model(inputs=inputs, outputs=x)

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])

# Print model summary
model.summary()

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

#Training Model
model.fit(X_train,y_train ,epochs=50,validation_data=(X_test,y_test),verbose=2, callbacks=[early_stop])

#Saving the model
filename="3DCNN_Noise_sig_10_1"
model.save("./models/"+filename+".h5",save_format='h5')



y_pred=model.predict(X_test)
y_true=y_test

# Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE): ", mse)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE): ", rmse)

# R-squared score (R2)
r2 = r2_score(y_true, y_pred)
print("R-squared score (R2): ", r2)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error (MAE): ", mae)

# Explained Variance Score (EVS)
evs = explained_variance_score(y_true, y_pred)
print("Explained Variance Score (EVS): ", evs)



# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')