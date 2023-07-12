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

cache_path = "/homes/siddharth/21cmFAST-cache/"


#File Constant
x=9
# Number of ionize box

n=int(args.no_of_sim)


# get the start time
st = time.time()

#Creating Lists
ion_list=[]
info_list=[]

for i in range(0,n):
    index=i
    red_shift=random.uniform(4, 9)
    seed= random.randint(10000, 9999999999)
    ionized_field = p21c.ionize_box(
    redshift = red_shift,
    user_params = {"HII_DIM": 128, "BOX_LEN": 100,"USE_INTERPOLATION_TABLES": True},
    astro_params = p21c.AstroParams({"HII_EFF_FACTOR":20.0}),
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.8),
    random_seed= seed
    )
    xh_box=np.array(ionized_field.xH_box)
    ion_list.append(xh_box)
    info_list.append([index,red_shift,seed])

    



ion_list=np.array(ion_list)
print(ion_list.shape)


fname='data'+args.no_of_sim+'_'+str(int(args.f_no)+(112*x))
folderpath='/tier2/tifr/darshan/testrun/rs_data/'+fname

#Creating Directory
os.mkdir(folderpath)

filepath= folderpath+'/'+fname

#Saving Dataset in H5 pickle format
hf = h5py.File(filepath+'.h5', 'w')
hf.create_dataset('ion_field', data=ion_list)

#Saving the parameters in CSV format
fields = ['Index', 'Redshift', 'Random Seed'] 
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
    file.write(' Execution time: '+str(elapsed_time)+' seconds')

cache_tools.clear_cache()