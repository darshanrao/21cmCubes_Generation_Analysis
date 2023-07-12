import numpy as np
import h5py
#import h5

# hf = h5py.File('./Data/Data_bt.h5', 'r')
# hf.keys()
# temp_box=hf.get('dataset')
# print(temp_box.shape)

zero_data=np.load('./Edge_cases_data/zero.npy')
one_data=np.load('./Edge_cases_data/one.npy')

zero_list=[]
for i in range(10):
    noise = 50.*np.random.randn(500,128,128,128)
    noise_data=zero_data+noise
    zero_list.append(list(noise_data))
zero_list=np.array(zero_list)
zero_list=np.reshape(zero_list,(5000,128,128,128))
print(zero_list.shape)

hf = h5py.File('./Noise_Data/sigma50_classes/zero.h5', 'w')
hf.create_dataset('dataset', data=zero_list)

one_list=[]
for i in range(10):
    noise = 50.*np.random.randn(500,128,128,128)
    noise_data=one_data+noise
    one_list.append(list(noise_data))
one_list=np.array(one_list)
one_list=np.reshape(one_list,(5000,128,128,128))
print(one_list.shape)

hf = h5py.File('./Noise_Data/sigma50_classes/one.h5', 'w')
hf.create_dataset('dataset', data=one_list)

# temp_box = temp_box+noise

# print(temp_box.shape)

# np.save('./Noise_Data/sigma10_classes/zero.npy',zero_list.astype(np.float32))
