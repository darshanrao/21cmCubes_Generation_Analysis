import py21cmfast as p21c
import os
import numpy as np
import h5py
import random
from py21cmfast import cache_tools
import csv
import time

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("no_of_sim",help="Num of generations")
parser.add_argument("f_no",help="File number")
args=parser.parse_args()

x=11
n=int(args.no_of_sim)


# get the start time
st = time.time()

#Creating Lists
ion_list=[]
temp_list=[]
info_list=[]

for i in range(0,n):
    index=i
    red_shift=random.uniform(4, 9)
    seed= random.randint(10000, 9999999999)

    initial_conditions = p21c.initial_conditions(
    user_params = {"HII_DIM": 128, "BOX_LEN": 100,"USE_INTERPOLATION_TABLES": True},
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8),
    random_seed=seed
    )
    perturbed_field = p21c.perturb_field(
    redshift = red_shift,
    init_boxes = initial_conditions
    )
    ionized_field = p21c.ionize_box(
    perturbed_field = perturbed_field,
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8),
    )
    brightness_temp=p21c.brightness_temperature(
    ionized_box=ionized_field,
    perturbed_field = perturbed_field,
    )
    temp_box=np.array(brightness_temp.brightness_temp)
    xh_box=np.array(ionized_field.xH_box)
    

    ion_list.append(xh_box)
    temp_list.append(temp_box)
    info_list.append([red_shift,seed])

    

ion_list=np.array(ion_list)
print(ion_list.shape)

temp_list=np.array(temp_list)
print(temp_list.shape)



fname='data'+args.no_of_sim+'_'+str(int(args.f_no)+(100*x))
folderpath='./Data/'+fname
#Creating Directory
os.mkdir(folderpath)
filepath= folderpath+'/'+fname

#Saving Dataset in H5 pickle format for Ionizatian Field
hf1 = h5py.File(filepath+'_if.h5', 'w')
hf1.create_dataset('ion_field', data=ion_list)

#Saving Dataset in H5 pickle format for Brightness Temperature
hf2 = h5py.File(filepath+'_bt.h5', 'w')
hf2.create_dataset('bright_temp', data=temp_list)

#Saving the parameters in CSV format
fields = ['Redshift', 'Random Seed'] 
with open(filepath+'.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(fields)
    csv_writer.writerows(info_list)


# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

with open(filepath+'.txt', "w") as file:
    file.write(str(ion_list.shape))
    file.write(str(temp_list.shape))
    file.write(' Execution time: '+str(elapsed_time)+' seconds')

cache_tools.clear_cache()