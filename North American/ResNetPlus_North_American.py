# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 06:11:38 2017

@author: CKJ
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

Input = np.genfromtxt('input.txt')
Output = np.genfromtxt('output.txt')

l = np.zeros((24*2842,))
t = np.zeros((24*2842,))

d_max = np.zeros((24*2842,))
d_min = np.zeros((24*2842,))
        
for i in range(np.shape(Output)[0]):
    for j in range(24):
        l[i*24 + j] = Output[i, 1+j]
    
for i in range(np.shape(Input)[0]):
    for j in range(24):
        t[i*24 + j] = Input[i, 1+j]

for i in range(24*2842):
    if l[i] == 0:
        l[i] = 0.5 * (l[i-1] + l[i+1])
    if t[i] <= 0:
        t[i] = 0.5 * (t[i-1] + t[i+1])

LOG = False

if LOG:
    l = (np.log(l)-6.8)/1.7
    t = (np.log(t)-3)/1.5

l_reshape = np.reshape(l,(2842,24))
for i in range(np.shape(Output)[0]):
    for j in range(24):
        d_max[i*24 + j] = max(l_reshape[i, :])
        d_min[i*24 + j] = min(l_reshape[i, :])

if not LOG:
    D = l / 5000
    T = t / 100
    D_max = d_max / 5000
    D_min = d_min / 5000
else:
    D = l
    T = t
    D_max = d_max
    D_min = d_min
    
iter_weekday = 2
weekday = np.zeros((24*2842,))
for i in range(2842):
    mod = np.mod(iter_weekday, 7)
    for j in range(24):
        if (mod == 6) or (mod == 0):
            weekday[i*24 + j] = 0
        else:
            weekday[i*24 + j] = 1
    iter_weekday += 1

import datetime
iter_date = datetime.date(1985, 1, 1)
season = np.zeros((24*2842,))
festival = np.zeros((24*2842,))
for i in range(2842):
    month = iter_date.month
    day = iter_date.day
    for j in range(24):
        if (month==4) | (month==5) | ((month==3) and (day>7)) | ((month==6) and (day<8)):
            season[i*24 + j] = 0
        elif (month==7) | (month==8) | ((month==6) and (day>7)) | ((month==9) and (day<8)):
            season[i*24 + j] = 1
        elif (month==10) | (month==11) | ((month==9) and (day>7)) | ((month==12) and (day<8)):
            season[i*24 + j] = 2
        elif (month==1) | (month==2) | ((month==12) and (day>7)) | ((month==3) and (day<8)):
            season[i*24 + j] = 3

        if (month == 7) and (day == 4):
            festival[i*24 + j] = 1
        if (month == 11) and (iter_date.weekday() == 4) and (day + 7 > 30):
            festival[i*24 + j] = 1
        if (month == 12) and (day == 25):
            festival[i*24 + j] = 1

    #print(str(i) + ' ' +str(month) + ' ' +str(day))

    iter_date = iter_date + datetime.timedelta(1)


def data_split(D, T, D_max, D_min, season, weekday, festival, train_split = 0.8, test_split = 0.2, validation_split = 0.1):
    x_1 = []
    x_21_D = []
    x_21_T = []
    x_22_D = []
    x_22_T = []
    x_23_D = []
    x_23_T = []
    x_3 = []
    x_4 = []
    x_5 = []
    x_season = []
    x_weekday = []
    x_festival = []
    y = []
    
    len_dataset = np.shape(D)[0]
    num_sample = len_dataset-8064
    
    for i in range(8064,len_dataset):    
        x_1.append(D[i-24:i])
        
        index_x_21 = [i-24, i-48, i-72, i-96, i-120, i-144, i-168]
        x_21_D.append(D[index_x_21])
        x_21_T.append(T[index_x_21])
        
        index_x_22 = [i-168, i-336, i-504, i-672, i-840, i-1008, i-1176, i-1344]
        x_22_D.append(D[index_x_22])
        x_22_T.append(T[index_x_22])
        
        index_x_23 = [i-672, i-1344, i-2016, i-2688, i-3360, i-4032, i-4704, i-5376, i-6048, i-6720, i-7392, i-8064]
        x_23_D.append(D[index_x_23])
        x_23_T.append(T[index_x_23])
        
        x_3.append(T[i])
        x_4.append(D_max[i])
        x_5.append(D_min[i])
        
        y.append(D[i])
    
        season_onehot = np.zeros(4)
        season_onehot[int(season[i])] = 1 
        x_season.append(season_onehot)

        weekday_onehot = np.zeros(2)
        weekday_onehot[int(weekday[i])] = 1 
        x_weekday.append(weekday_onehot)

        festival_onehot = np.zeros(2)
        festival_onehot[int(festival[i])] = 1 
        x_festival.append(festival_onehot)
        
    #X_1 = np.reshape(np.array(x_1), (np.shape(np.array(x_1))[0], np.shape(np.array(x_1))[1], 1))
    X_1 = np.array(x_1)
    X_21_D = np.array(x_21_D)
    X_21_T = np.array(x_21_T)
    X_22_D = np.array(x_22_D)
    X_22_T = np.array(x_22_T)
    X_23_D = np.array(x_23_D)
    X_23_T = np.array(x_23_T)
    X_3 = np.array(x_3)
    X_4 = np.array(x_4)
    X_5 = np.array(x_5)
    X_season = np.array(x_season)
    X_weekday = np.array(x_weekday)
    X_festival = np.array(x_festival)
    Y_1 = np.array(y)
    
    #num_train = int(num_sample * train_split)
    #num_val = int(num_sample * val_split)
    #8*365=2920，training:3000，test:the remainder
    num_train = (1100) * 24
    num_val = int(num_train * validation_split)
    
    X_train = []
    X_val = []
    X_test = []
    Y_train = []
    Y_val = []
    Y_test = []
    for i in range(24):
        #               0                          1                         2                         3                         4                         5                         6                         7                    8                    9                    10                             11                             12                             13                              14                              15                              16                         17                            18                            19                    
        X_train.append([X_1[i:num_train:24,:24-i], X_21_D[i:num_train:24,:], X_21_T[i:num_train:24,:], X_22_D[i:num_train:24,:], X_22_T[i:num_train:24,:], X_23_D[i:num_train:24,:], X_23_T[i:num_train:24,:], X_3[i:num_train:24], X_4[i:num_train:24], X_5[i:num_train:24], X_season[i:num_train:24,:], X_weekday[i:num_train:24,:], X_festival[i:num_train:24,:]])
        X_val.append([X_1[num_train-num_val+i:num_train:24,:24-i], X_21_D[num_train-num_val+i:num_train:24,:], X_21_T[num_train-num_val+i:num_train:24,:], X_22_D[num_train-num_val+i:num_train:24,:], X_22_T[num_train-num_val+i:num_train:24,:], X_23_D[num_train-num_val+i:num_train:24,:], X_23_T[num_train-num_val+i:num_train:24,:], X_3[num_train-num_val+i:num_train:24], X_4[num_train-num_val+i:num_train:24], X_5[num_train-num_val+i:num_train:24], X_season[num_train-num_val+i:num_train:24,:], X_weekday[num_train-num_val+i:num_train:24,:], X_festival[num_train-num_val+i:num_train:24,:]])
        X_test.append([X_1[num_train+i:num_sample:24,:24-i], X_21_D[num_train+i:num_sample:24,:], X_21_T[num_train+i:num_sample:24,:], X_22_D[num_train+i:num_sample:24,:], X_22_T[num_train+i:num_sample:24,:], X_23_D[num_train+i:num_sample:24,:], X_23_T[num_train+i:num_sample:24,:], X_3[num_train+i:num_sample:24], X_4[num_train+i:num_sample:24], X_5[num_train+i:num_sample:24], X_season[num_train+i:num_sample:24,:], X_weekday[num_train+i:num_sample:24,:], X_festival[num_train+i:num_sample:24,:]])
        Y_train.append(Y_1[i:num_train:24])
        Y_val.append(Y_1[num_train-num_val+i:num_train:24])
        Y_test.append(Y_1[num_train+i:num_sample:24])

    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

# test : 286+365 = 651 days
num_pre_days = 336
num_train = 1100
num_days = num_train + num_pre_days + 650
num_data_points = num_days * 24
num_days_start = 2842-num_pre_days-650-num_train
start_data_point = num_days_start * 24
X_train, X_val, X_test, Y_train, Y_val, Y_test = data_split(D[start_data_point: start_data_point + num_data_points], T[start_data_point: start_data_point + num_data_points], D_max[start_data_point: start_data_point + num_data_points], D_min[start_data_point: start_data_point + num_data_points], season[start_data_point: start_data_point + num_data_points], weekday[start_data_point: start_data_point + num_data_points], festival[start_data_point: start_data_point + num_data_points], 0.8, 0.2, 0.1)

## ----------------------------------------------------------------------------
from keras.models import Model
from keras.layers import Input, Dense, LSTM, concatenate, Activation, add, BatchNormalization
from keras.layers.merge import multiply, maximum, dot, average
from keras import backend as K
from keras.losses import mean_absolute_percentage_error, hinge
from keras.regularizers import l1, l2
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping 
from keras.optimizers import SGD, adam
from keras.layers.advanced_activations import PReLU, ELU

def get_input(hour):
    input_Dd = Input(shape=(7,), name = 'input'+str(hour)+'_Dd')
    input_Dw = Input(shape=(8,), name = 'input'+str(hour)+'_Dw')
    input_Dm = Input(shape=(12,), name = 'input'+str(hour)+'_Dm')
    input_Dr = Input(shape=(24-hour+1,), name = 'input'+str(hour)+'_Dr')
    
    input_Td = Input(shape=(7,), name = 'input'+str(hour)+'_Td')
    input_Tw = Input(shape=(8,), name = 'input'+str(hour)+'_Tw')
    input_Tm = Input(shape=(12,), name = 'input'+str(hour)+'_Tm')
    
    input_T = Input(shape=(1,))
    
    return (input_Dd, input_Dw, input_Dm, input_Dr, input_Td, input_Tw, input_Tm, input_T)
    
input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T = get_input(1)
input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T = get_input(2)
input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T = get_input(3)
input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T = get_input(4)
input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T = get_input(5)
input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T = get_input(6)
input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T = get_input(7)
input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T = get_input(8)
input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T = get_input(9)
input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T = get_input(10)
input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T = get_input(11)
input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T = get_input(12)
input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T = get_input(13)
input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T = get_input(14)
input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T = get_input(15)
input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T = get_input(16)
input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T = get_input(17)
input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T = get_input(18)
input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T = get_input(19)
input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T = get_input(20)
input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T = get_input(21)
input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T = get_input(22)
input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T = get_input(23)
input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T = get_input(24)
input_D_max = Input(shape=(1,), name = 'input_D_max')
input_D_min = Input(shape=(1,), name = 'input_D_min')
input_season = Input(shape=(4,), name = 'input_season')
input_weekday = Input(shape=(2,), name = 'input_weekday')
input_festival = Input(shape=(2,), name = 'input_festival')

num_dense1 = 10
num_dense2 = 10
num_dense1_Dr = 10 

DenseConcatMid = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal')
DenseConcatDr = Dense(num_dense1_Dr, activation = 'selu', kernel_initializer = 'lecun_normal')
DenseMid = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal')
DenseConcat1 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal')
DenseConcat2 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal')
DenseConcat3 = Dense(num_dense2, activation = 'selu', kernel_initializer = 'lecun_normal')

def get_layer_1(hour, input_Dd, input_Dw, input_Dm, input_Dr, input_Td, input_Tw, input_Tm, input_T, output_pre=[]):
    
    dense_Dd = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Dd)
    dense_Dw = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Dw)
    dense_Dm = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Dm)
    dense_Td = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Td)
    dense_Tw = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Tw)
    dense_Tm = Dense(num_dense1, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Tm)
    
    '''
    dense_Dd = ELU()(dense_Dd)
    dense_Dw = ELU()(dense_Dw)
    dense_Dm = ELU()(dense_Dm)
    dense_Td = ELU()(dense_Td)
    dense_Tw = ELU()(dense_Tw)
    dense_Tm = ELU()(dense_Tm)
    '''
    
    concat1 = concatenate([dense_Dd, dense_Td])
    dense_concat1 = DenseConcat1(concat1)
    #dense_concat1 = ELU()(dense_concat1)
    #dense1 = Dense(1, activation = 'sigmoid')(dense_concat1)
    
    concat2 = concatenate([dense_Dw, dense_Tw])
    dense_concat2 = DenseConcat2(concat2)
    #dense_concat2 = ELU()(dense_concat2)
    #dense2 = Dense(1, activation = 'sigmoid')(dense_concat2)
    
    concat3 = concatenate([dense_Dm, dense_Tm])
    dense_concat3 = DenseConcat3(concat3)
    #dense_concat3 = ELU()(dense_concat3)
    #dense3 = Dense(1, activation = 'sigmoid')(dense_concat3)
    
    concat_date_info = concatenate([input_season, input_weekday])
    dense_concat_date_info_1 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
    dense_concat_date_info_2 = Dense(5, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_date_info)
    
    concat_mid = concatenate([dense_concat1, dense_concat2, dense_concat3, dense_concat_date_info_1, input_festival])
    dense_concat_mid = DenseConcatMid(concat_mid)
    #dense_concat_mid = ELU()(dense_concat_mid)
    
    if output_pre == []:
        dense_Dr = Dense(num_dense1_Dr, activation = 'selu', kernel_initializer = 'lecun_normal')(input_Dr)
        #dense_Dr = ELU()(dense_Dr)
    else:
        concat_Dr = concatenate([input_Dr] + output_pre)
        dense_Dr = Dense(num_dense1_Dr, activation = 'selu', kernel_initializer = 'lecun_normal')(concat_Dr)
        #dense_Dr = ELU()(dense_Dr)
    dense_4 = DenseConcatDr(concatenate([dense_Dr, dense_concat_date_info_2]))
    #dense_4 = ELU()(dense_4)
        
    concat = concatenate([dense_concat_mid, dense_4, input_T])
    dense_mid = DenseMid(concat)
    #dense_mid = ELU()(dense_mid)
    
    output = Dense(1, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_mid)
    
    output_pre_new = output_pre + [output]
    return (output, output_pre_new)
      
output1, output_pre1 = get_layer_1(1, input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T)
output2, output_pre2 = get_layer_1(2, input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T, output_pre1)
output3, output_pre3 = get_layer_1(3, input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T, output_pre2)
output4, output_pre4 = get_layer_1(4, input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T, output_pre3)
output5, output_pre5 = get_layer_1(5, input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T, output_pre4)
output6, output_pre6 = get_layer_1(6, input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T, output_pre5)
output7, output_pre7 = get_layer_1(7, input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T, output_pre6)
output8, output_pre8 = get_layer_1(8, input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T, output_pre7)
output9, output_pre9 = get_layer_1(9, input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T, output_pre8)
output10, output_pre10 = get_layer_1(10, input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T, output_pre9)
output11, output_pre11 = get_layer_1(11, input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T, output_pre10)
output12, output_pre12 = get_layer_1(12, input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T, output_pre11)
output13, output_pre13 = get_layer_1(13, input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T, output_pre12)
output14, output_pre14 = get_layer_1(14, input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T, output_pre13)
output15, output_pre15 = get_layer_1(15, input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T, output_pre14)
output16, output_pre16 = get_layer_1(16, input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T, output_pre15)
output17, output_pre17 = get_layer_1(17, input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T, output_pre16)
output18, output_pre18 = get_layer_1(18, input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T, output_pre17)
output19, output_pre19 = get_layer_1(19, input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T, output_pre18)
output20, output_pre20 = get_layer_1(20, input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T, output_pre19)
output21, output_pre21 = get_layer_1(21, input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T, output_pre20)
output22, output_pre22 = get_layer_1(22, input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T, output_pre21)
output23, output_pre23 = get_layer_1(23, input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T, output_pre22)
output24, output_pre24 = get_layer_1(24, input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T, output_pre23)

def get_res_layer(output, last=False):
    dense_res11 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal')(output)
    dense_res12 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res11)
    
    dense_res21 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal')(output)
    dense_res22 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res21)
    '''
    dense_res31 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal')(output)
    dense_res32 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res31)

    dense_res41 = Dense(20, activation = 'selu', kernel_initializer = 'lecun_normal')(output)
    dense_res42 = Dense(24, activation = 'linear', kernel_initializer = 'lecun_normal')(dense_res41)
    '''
    #dense_add = add([dense_res12, dense_res22, dense_res32, dense_res42])
    dense_add = add([dense_res12, dense_res22])
    
    if last:
        output_new = add([dense_add, output], name = 'output')
    else:
        output_new = add([dense_add, output])
    return output_new

output_pre = concatenate(output_pre24)

def resnetplus_layer(input_1, input_2, output_list):
    output_res = get_res_layer(input_1)
    output_res_ = get_res_layer(input_2)
    output_res_ave_mid = average([output_res, output_res_])
    output_list.append(output_res_ave_mid)
    output_res_ave = average(output_list)
    return output_res_ave, output_list
    
input_1 = output_pre
input_2 = output_pre
output_list = [output_pre]

num_resnetplus_layer = 30

for i in range(num_resnetplus_layer):
    output_res_ave, output_list = resnetplus_layer(input_1, input_2, output_list)
    input_1 = output_res_ave
    if i == 0:
        input_2 = output_res_ave

output = output_res_ave

def penalized_loss(y_true, y_pred):
    beta = 0.5
    loss1 = mean_absolute_percentage_error(y_true, y_pred)
    loss2 = K.mean(K.maximum(K.max(y_pred, axis=1) - input_D_max, 0.), axis=-1)
    loss3 = K.mean(K.maximum(input_D_min - K.min(y_pred, axis=1), 0.), axis=-1)
    return loss1 + beta * (loss2 + loss3)

#model.compile(optimizer = 'rmsprop', loss = 'mape')
#model.compile(optimizer = 'adam', loss = penalized_loss)

def get_XY(X, Y):
    X_new = []
    Y_new = []
    for i in range(24):
        X_new.append(X[i][1])
        X_new.append(X[i][3])
        X_new.append(X[i][5])
        X_new.append(X[i][0])
        X_new.append(X[i][2])
        X_new.append(X[i][4])
        X_new.append(X[i][6])
        X_new.append(X[i][7]) # temperature
        Y_new.append(Y[i])
    X_new = X_new + [X[0][8],X[0][9],X[0][10],X[0][11],X[0][12]] # for shared input, use the data of hour 1 is enough.
    Y_new = [np.squeeze(np.array(Y_new)).transpose()] # the aggregate output of 24 single outputs
    #Y_new = [np.squeeze(np.array(Y_new)).transpose()] * 2 # aggregate twice!
    return (X_new, Y_new)

# -----------------------------------------------------------------------------

X_train_fit, Y_train_fit = get_XY(X_train, Y_train)
X_test_pred, Y_test_pred = get_XY(X_test, Y_test)

def get_model():
    model = Model(inputs=[input1_Dd, input1_Dw, input1_Dm, input1_Dr, input1_Td, input1_Tw, input1_Tm, input1_T,\
                      input2_Dd, input2_Dw, input2_Dm, input2_Dr, input2_Td, input2_Tw, input2_Tm, input2_T,\
                      input3_Dd, input3_Dw, input3_Dm, input3_Dr, input3_Td, input3_Tw, input3_Tm, input3_T,\
                      input4_Dd, input4_Dw, input4_Dm, input4_Dr, input4_Td, input4_Tw, input4_Tm, input4_T,\
                      input5_Dd, input5_Dw, input5_Dm, input5_Dr, input5_Td, input5_Tw, input5_Tm, input5_T,\
                      input6_Dd, input6_Dw, input6_Dm, input6_Dr, input6_Td, input6_Tw, input6_Tm, input6_T,\
                      input7_Dd, input7_Dw, input7_Dm, input7_Dr, input7_Td, input7_Tw, input7_Tm, input7_T,\
                      input8_Dd, input8_Dw, input8_Dm, input8_Dr, input8_Td, input8_Tw, input8_Tm, input8_T,\
                      input9_Dd, input9_Dw, input9_Dm, input9_Dr, input9_Td, input9_Tw, input9_Tm, input9_T,\
                      input10_Dd, input10_Dw, input10_Dm, input10_Dr, input10_Td, input10_Tw, input10_Tm, input10_T,\
                      input11_Dd, input11_Dw, input11_Dm, input11_Dr, input11_Td, input11_Tw, input11_Tm, input11_T,\
                      input12_Dd, input12_Dw, input12_Dm, input12_Dr, input12_Td, input12_Tw, input12_Tm, input12_T,\
                      input13_Dd, input13_Dw, input13_Dm, input13_Dr, input13_Td, input13_Tw, input13_Tm, input13_T,\
                      input14_Dd, input14_Dw, input14_Dm, input14_Dr, input14_Td, input14_Tw, input14_Tm, input14_T,\
                      input15_Dd, input15_Dw, input15_Dm, input15_Dr, input15_Td, input15_Tw, input15_Tm, input15_T,\
                      input16_Dd, input16_Dw, input16_Dm, input16_Dr, input16_Td, input16_Tw, input16_Tm, input16_T,\
                      input17_Dd, input17_Dw, input17_Dm, input17_Dr, input17_Td, input17_Tw, input17_Tm, input17_T,\
                      input18_Dd, input18_Dw, input18_Dm, input18_Dr, input18_Td, input18_Tw, input18_Tm, input18_T,\
                      input19_Dd, input19_Dw, input19_Dm, input19_Dr, input19_Td, input19_Tw, input19_Tm, input19_T,\
                      input20_Dd, input20_Dw, input20_Dm, input20_Dr, input20_Td, input20_Tw, input20_Tm, input20_T,\
                      input21_Dd, input21_Dw, input21_Dm, input21_Dr, input21_Td, input21_Tw, input21_Tm, input21_T,\
                      input22_Dd, input22_Dw, input22_Dm, input22_Dr, input22_Td, input22_Tw, input22_Tm, input22_T,\
                      input23_Dd, input23_Dw, input23_Dm, input23_Dr, input23_Td, input23_Tw, input23_Tm, input23_T,\
                      input24_Dd, input24_Dw, input24_Dm, input24_Dr, input24_Td, input24_Tw, input24_Tm, input24_T,\
                      input_D_max, input_D_min, input_season, input_weekday, input_festival], \
                      outputs=[output])
    return model
      
model = get_model()
model.compile(optimizer = 'adam', loss = penalized_loss)
model.save_weights('model.h5')
 
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

mape_list = []
history_list = []
pred_list = []

from keras.callbacks import LearningRateScheduler
def lr_scheduler1(epoch, mode=None):
    lr = 0.001
    return lr

def lr_scheduler2(epoch, mode=None):
    lr = 0.0005
    return lr

def lr_scheduler3(epoch, mode=None):
    lr = 0.0003
    return lr

scheduler1 = LearningRateScheduler(lr_scheduler1)
scheduler2 = LearningRateScheduler(lr_scheduler2)
scheduler3 = LearningRateScheduler(lr_scheduler3)

num_repeat = 8
NUM_TEST = 650
BATCH_SIZE = 32
NUM_SNAPSHOT = 18
for i in range(num_repeat):
    model.load_weights('model.h5')
    shuffle_weights(model)
    
    history_1 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=500, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'1_weights.h5')    
    print(str(i) + ' 1')
    
    history_2 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'2_weights.h5') 
    print(str(i) + ' 2')
    
    history_3 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'3_weights.h5') 
    print(str(i) + ' 3')  
    
    history_4 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'4_weights.h5') 
    print(str(i) + ' 4')  
        
    history_5 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'5_weights.h5') 
    print(str(i) + ' 5')  
        
    history_6 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'6_weights.h5')    
    print(str(i) + ' 6')  
        
    history_7 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'7_weights.h5')    
    print(str(i) + ' 7')  
        
    history_8 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'8_weights.h5')    
    print(str(i) + ' 8')  
        
    history_9 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'9_weights.h5')    
    print(str(i) + ' 9')  
        
    history_10 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'10_weights.h5')    
    print(str(i) + ' 10')  
    
    history_11 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler1])
    
    model.save_weights('complete' + str(i+1) +'11_weights.h5')   
    print(str(i) + ' 11')  
        
    history_12 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'12_weights.h5')    
    print(str(i) + ' 12')  
        
    history_13 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'13_weights.h5')    
    print(str(i) + ' 13')  
        
    history_14 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'14_weights.h5')    
    print(str(i) + ' 14')  
    
    history_15 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler2])
    
    model.save_weights('complete' + str(i+1) +'15_weights.h5')    
    print(str(i) + ' 15')  
    
    history_16 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'16_weights.h5')    
    print(str(i) + ' 16')  
    
    history_17 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'17_weights.h5')    
    print(str(i) + ' 17')  
    
    history_18 = model.fit(X_train_fit, Y_train_fit, \
                        epochs=50, batch_size=BATCH_SIZE, callbacks=[scheduler3])
    
    model.save_weights('complete' + str(i+1) +'18_weights.h5')    
    print(str(i) + ' 18')  
    
    history_list.append([history_1, history_2, history_3, history_4, history_5, history_6, history_7, history_8, history_9, history_10, history_11, history_12, history_13, history_14, history_15, history_16, history_17, history_18])
'''
def get_curve_data(history):
    loss = []
    val_loss = []
    for history_item in history:
        loss = loss + history_item.history['loss']
        val_loss = val_loss + history_item.history['val_loss']
    return (np.array(loss), np.array(val_loss))

loss_list = []
val_loss_list = []
for history in history_list:
    loss_once, val_loss_once = get_curve_data(history)
    loss_list.append(loss_once)
    val_loss_list.append(val_loss_once)
    
loss = np.array(loss_list)
val_loss = np.array(val_loss_list)

loss_mean = np.mean(loss, axis = 0)
loss_std = np.std(loss, axis = 0)
loss_up = loss_mean + loss_std
loss_down = loss_mean - loss_std

val_loss_mean = np.mean(val_loss, axis = 0)
val_loss_std = np.std(val_loss, axis = 0)
val_loss_up = val_loss_mean + val_loss_std
val_loss_down = val_loss_mean - val_loss_std

x = range(1350)

plt.figure(figsize = (5,4))
plt.plot(x, loss_mean, color = 'Green')
plt.fill_between(x, loss_up, loss_down, color='LightGreen', alpha=0.7) 
plt.plot(val_loss_mean, color = 'RoyalBlue')
plt.fill_between(x, val_loss_up, val_loss_down, color='LightSkyBlue', alpha=0.7) 
plt.axis([0, 1350, 1, 10])
'''

loss = np.zeros((NUM_SNAPSHOT, NUM_SNAPSHOT))
for i in tqdm(range(0, NUM_SNAPSHOT)):
    for j in range(i, NUM_SNAPSHOT):
        p = np.zeros((num_repeat*(j-i+1),24*NUM_TEST))
        for k in range(num_repeat):
            for l in range(i,j+1):
                model.load_weights('complete' + str(k+1) + str(l+1) + '_weights.h5')
                pred = model.predict(X_test_pred)
                p[k*(j-i+1)+l-i,:] = pred.reshape(24*NUM_TEST) 
        pred_eval = np.mean(p, axis = 0)
        Y_test_eval = np.array(Y_test).transpose().reshape(24*NUM_TEST)
        mape = np.mean(np.divide(np.abs(Y_test_eval - pred_eval), Y_test_eval))
        loss[i,j] = mape