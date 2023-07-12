import numpy as np
import h5py
#import h5
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import time

from tensorflow.keras.callbacks import EarlyStopping

# get the start time
st = time.time()

# hf = h5py.File('./Noise_Data/sigma50_classes/one.h5', 'r')
# hf.keys()
# one_data=hf.get('dataset')
# print(one_data.shape)
# hf = h5py.File('./Noise_Data/sigma50_classes/zero.h5', 'r')
# hf.keys()
# zero_data=hf.get('dataset')
# print(zero_data.shape)
# X=np.concatenate((np.array(zero_data),np.array(one_data)))
# np.save('./Noise_Data/sigma50_classes/data.npy',X.astype(np.float32))



y=np.concatenate((np.zeros(5000),np.ones(5000)))
X=np.load('./Noise_Data/sigma100_classes/data.npy')
print(X.shape)

def normalized(arr):
    if np.std(arr) != 0:
        arr=(arr - np.mean(arr)) / np.std(arr)
    return arr

for i in range(0,10000):
    X[i]=normalized(X[i])

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X_train.reshape(X_train.shape[0],128,128,128,1)
X_test = X_test.reshape(X_test.shape[0],128,128,128,1)


# Define input shape
input_shape = (128, 128, 128, 1)

# Define input layer
inputs = Input(shape=input_shape)


# Define convolutional layers
x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(inputs)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
x = Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)
# Define flatten layer
x = Flatten()(x)
# Define fully connected layers
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1,activation='sigmoid')(x)

# Define model
model = Model(inputs=inputs, outputs=x)


# Adam = tf.keras.optimizers.Adam(learning_rate=5e-7)
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

# Print model summary
model.summary()

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

#Training Model
model.fit(X_train,y_train ,epochs=50,validation_data=(X_test,y_test),verbose=2,callbacks=[early_stop])

#Saving the model
filename="3DCNN_Class_sigma100_2"
model.save("./models/"+filename+".h5",save_format='h5')

y_pred= model.predict(X_test)
y_labels= [int(np.around(element)) for element in y_pred]

from sklearn.metrics import confusion_matrix,accuracy_score
acc= accuracy_score(y_test, y_labels)
cm=confusion_matrix(y_test,y_labels)
print(cm)
print(acc)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')











