import setGPU
import os
import pandas as pd
import numpy as np
import nibabel as nib
import glob
import math
import random
import time
import os
from os.path import join
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential,Model
#from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
import keras
import keras.backend as K

n_epochs = 50
lr = 27*1e-6
VALIDATION_SPLIT = 0.2
List_con = ["Age","FIQ"]
Min_Couples = 20
N_Attempts = 50
Matching_diff = [3,15]

def Get_phenotypicCharacteristics():
    df_NYU = pd.read_csv("/home/phenotypic_NYU.csv")
    df_KKI = pd.read_csv("/home/ABIDEII-KKI_1.csv")
    df_KKI_Add = pd.read_csv("/home/ABIDE-II_KKI_1_AdditionalScanInfo.csv")
    phenotypicCharacteristics = {}
    for i in range(len(df_NYU)):
        path = "/home/NYU_crop_ds/00{}_crop_ds.nii.gz".format(df_NYU['SUB_ID'][i])
        phenotypicCharacteristics.update({ df_NYU['SUB_ID'][i] : {'Site' : df_NYU['SITE_ID'][i], 'Sex' : df_NYU['SEX'][i], 'Diagnostic category' : df_NYU['DX_GROUP'][i], 'Age' : "{:.0f}".format(df_NYU['AGE_AT_SCAN'][i]), 'FIQ' : "{:.0f}".format(df_NYU['FIQ'][i]), 'Path' : path}})
    for i in range(len(df_KKI)):
        site = "{}_{}".format(df_KKI['SITE_ID'][i],df_KKI_Add['RECEIVING_COIL'][i]).replace(' ', '_')
        path = "/home/KKI_crop_ds/sub-{}_crop_ds.nii.gz".format(df_KKI['SUB_ID'][i])
        phenotypicCharacteristics.update({ df_KKI['SUB_ID'][i] : {'Site' : site, 'Sex' : df_KKI['SEX'][i], 'Diagnostic category' : df_KKI['DX_GROUP'][i], 'Age' : "{:.0f}".format(df_KKI['AGE_AT_SCAN '][i]), 'FIQ' : "{:.0f}".format(df_KKI['FIQ'][i]), 'Path' : path}})
    return phenotypicCharacteristics

def Matching(cur_con,c_con,Matching_diff):
    for idx in range(len(List_con)):
        #print(cur_con[idx],c_con[idx])
        if (cur_con[idx]=='nan')or(c_con[idx]=='nan')or(abs(int(cur_con[idx]) - int(c_con[idx])) >= Matching_diff[idx]):
            return False
    return True

def Matching_Function(phenotypicCharacteristics,List_1,List_2,List_con,Matching_diff,N_Matched_Lists,Min_Couples,N_Attempts):

    while(N_Attempts>0):
        List_1 = np.asarray(List_1)
        List_2 = np.asarray(List_2)
        l_1st = (List_1.copy() if len(List_1) <= len(List_2) else List_2.copy()).tolist()
        l_2nd = (List_2.copy() if len(List_2) >= len(List_1) else List_1.copy()).tolist()
        N_Attempts = N_Attempts-1
        result = list()
        for i in range(len(l_1st)):
            cur_con = list()
            for idx in range(len(List_con)):
                cur_con.append(phenotypicCharacteristics.get(l_1st[i])[List_con[idx]])
            candidate = list()
            if len(l_2nd) > 0:
                for j in range(len(l_2nd)):
                    c_con = list()
                    for idx in range(len(List_con)):
                        c_con.append(phenotypicCharacteristics.get(l_2nd[j])[List_con[idx]])
                    if Matching(cur_con,c_con,Matching_diff):
                        candidate.append(l_2nd[j])
                #print(candidate)
                if len(candidate) > 0:
                    idx = random.choice(range(len(candidate)))
                    #print(len(candidate),idx,candidate[idx])
                    result.append(tuple((l_1st[i],candidate[idx])))
                    l_2nd.remove(candidate[idx])
            else:
                break
        #print(len(result))
        if (result not in N_Matched_Lists)&(len(result)!=0)&(len(result)>=Min_Couples):
            N_Matched_Lists.append(result)

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st Volumetric Convolutional block
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=(110, 110, 110, 1)))
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 2nd Volumetric Convolutional block
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 3rd Volumetric Convolutional block
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 4th Volumetric Convolutional block
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # 1th Deconvolutional layer with batchnorm and dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    # 2th Deconvolutional layer
    model.add(Dense(64, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    # Output with softmax nonlinearity for classification
    model.add(Dense(2, activation='softmax'))
    if summary:
        print(model.summary())
    return model

def get_dataset():

    phenotypicCharacteristics = Get_phenotypicCharacteristics()

    M_NYU_Candidates = [ key for key,val in phenotypicCharacteristics.items() if (val['Sex']==1)&(val['Site']=='NYU')&(val['Diagnostic category']==2) ]
    F_NYU_Candidates = [ key for key,val in phenotypicCharacteristics.items() if (val['Sex']==2)&(val['Site']=='NYU')&(val['Diagnostic category']==2) ]
    M_KKI_8_Candidates = [ key for key,val in phenotypicCharacteristics.items() if (val['Sex']==1)&(val['Site']=='ABIDEII-KKI_1_8_channel')&(val['Diagnostic category']==2)]
    F_KKI_8_Candidates = [ key for key,val in phenotypicCharacteristics.items() if (val['Sex']==2)&(val['Site']=='ABIDEII-KKI_1_8_channel')&(val['Diagnostic category']==2)]
    #print(len(M_NYU_Candidates),len(F_NYU_Candidates),len(M_KKI_8_Candidates),len(F_KKI_8_Candidates))
    
    NYU_Matched_Lists = list()
    Matching_Function(phenotypicCharacteristics,M_NYU_Candidates,F_NYU_Candidates,List_con,Matching_diff,NYU_Matched_Lists,Min_Couples,N_Attempts)
    KKI_8_Matched_Lists = list()
    Matching_Function(phenotypicCharacteristics,M_KKI_8_Candidates,F_KKI_8_Candidates,List_con,Matching_diff,KKI_8_Matched_Lists,Min_Couples,N_Attempts)

    F_NYU_Matched = np.asarray(max(NYU_Matched_Lists, key=len))[:,0].tolist()
    M_NYU_Matched = np.asarray(max(NYU_Matched_Lists, key=len))[:,1].tolist()
    F_KKI_8_Matched = np.asarray(max(KKI_8_Matched_Lists, key=len))[:,0].tolist()
    M_KKI_8_Matched = np.asarray(max(KKI_8_Matched_Lists, key=len))[:,1].tolist()
    #print(len(M_NYU_Matched),len(F_NYU_Matched),len(F_KKI_8_Matched),len(M_KKI_8_Matched))

    N_samples_n = 8
    N_VAL = 8

    Indices_NYU = np.random.choice(len(M_NYU_Matched), size = N_samples_n, replace=False)
    Indices_KKI = np.random.choice(len(F_KKI_8_Matched), size = N_samples_n, replace=False)

    M_NYU_Chosen = [item for item in M_NYU_Matched if ((M_NYU_Matched.index(item) in Indices_NYU))]
    F_NYU_Chosen = [item for item in F_NYU_Matched if ((F_NYU_Matched.index(item) in Indices_NYU))]
    F_KKI_8_Chosen = [item for item in F_KKI_8_Matched if ((F_KKI_8_Matched.index(item) in Indices_KKI))]
    M_KKI_8_Chosen = [item for item in M_KKI_8_Matched if ((M_KKI_8_Matched.index(item) in Indices_KKI))]
    #print(len(M_NYU_Chosen),len(F_NYU_Chosen),len(F_KKI_8_Chosen),len(M_KKI_8_Chosen))

    M_NYU_Rest = np.setdiff1d(M_NYU_Candidates,M_NYU_Chosen).tolist()
    F_KKI_8_Rest = np.setdiff1d(F_KKI_8_Candidates,F_KKI_8_Chosen).tolist()
    #print(len(M_NYU_Candidates),len(M_NYU_Rest),len(F_KKI_8_Candidates),len(F_KKI_8_Rest))

    Rest_Matched_Lists = list()
    Matching_Function(phenotypicCharacteristics,M_NYU_Rest,F_KKI_8_Rest,List_con,Matching_diff,Rest_Matched_Lists,Min_Couples,N_Attempts)
    M_NYU_Rest_matched = np.asarray(max(Rest_Matched_Lists, key=len))[:,1].tolist()
    F_KKI_8_Rest_matched = np.asarray(max(Rest_Matched_Lists, key=len))[:,0].tolist()
    #print(len(M_NYU_Rest_matched),len(F_KKI_8_Rest_matched))

    M_NYU_TR = np.append(M_NYU_Chosen,M_NYU_Rest_matched)
    F_KKI_8_TR = np.append(F_KKI_8_Chosen,F_KKI_8_Rest_matched)
    F_NYU_TR = F_NYU_Chosen
    M_KKI_8_TR = M_KKI_8_Chosen
    #print(len(M_NYU_TR),len(F_KKI_8_TR),len(F_NYU_TR),len(M_KKI_8_TR))

    F_NYU_VAL_Candidates = np.setdiff1d(F_NYU_Candidates,F_NYU_Chosen).tolist()
    M_KKI_8_VAL_Candidates = np.setdiff1d(M_KKI_8_Candidates,M_KKI_8_Chosen).tolist()
    #print(len(F_NYU_VAL_Candidates),len(M_KKI_8_VAL_Candidates))

    Indices_F_NYU_VAL = np.random.choice(len(F_NYU_VAL_Candidates), size = N_VAL, replace=False)
    Indices_M_KKI_VAL = np.random.choice(len(M_KKI_8_VAL_Candidates), size = N_VAL, replace=False)

    F_NYU_VAL = [item for item in F_NYU_VAL_Candidates if ((F_NYU_VAL_Candidates.index(item) in Indices_F_NYU_VAL))]
    M_KKI_8_VAL = [item for item in M_KKI_8_VAL_Candidates if ((M_KKI_8_VAL_Candidates.index(item) in Indices_M_KKI_VAL))]
    #print(len(F_NYU_VAL),len(M_KKI_8_VAL))

    TR_1_NYU = M_NYU_TR
    TR_1_KKI = M_KKI_8_TR
    TR_2_NYU = F_NYU_TR
    TR_2_KKI = F_KKI_8_TR
    VAL_1_KKI = M_KKI_8_VAL
    VAL_2_NYU = F_NYU_VAL   
    #print(len(TR_1_NYU),len(TR_2_NYU),len(TR_1_KKI),len(TR_2_KKI))
    #print(len(VAL_1_KKI),len(VAL_2_NYU))

    fps_TR_set = list()
    fps_VAL_set = list()
    TR_set = list()
    VAL_set = list()
    L_TR = list()
    L_VAL = list()
    Lb_TR = list()
    Lb_VAL = list()

    for idx in range(len(TR_1_NYU)):
        fps_TR_set.append(phenotypicCharacteristics.get(TR_1_NYU[idx])['Path'])
        L_TR.append([0,1])
        Lb_TR.append([0,1]) 
    for idx in range(len(TR_1_KKI)):
        fps_TR_set.append(phenotypicCharacteristics.get(TR_1_KKI[idx])['Path'])
        L_TR.append([0,1])
        Lb_TR.append([1,0])
    for idx in range(len(TR_2_NYU)):
        fps_TR_set.append(phenotypicCharacteristics.get(TR_2_NYU[idx])['Path'])
        L_TR.append([1,0])
        Lb_TR.append([0,1])
    for idx in range(len(TR_2_KKI)):
        fps_TR_set.append(phenotypicCharacteristics.get(TR_2_KKI[idx])['Path'])
        L_TR.append([1,0])
        Lb_TR.append([1,0])

    for idx in range(len(VAL_1_KKI)):
        fps_VAL_set.append(phenotypicCharacteristics.get(VAL_1_KKI[idx])['Path'])
        L_VAL.append([0,1])
        Lb_VAL.append([1,0])
    for idx in range(len(VAL_2_NYU)):
        fps_VAL_set.append(phenotypicCharacteristics.get(VAL_2_NYU[idx])['Path'])
        L_VAL.append([1,0])
        Lb_VAL.append([0,1])
    
    Indices_VAL = np.random.choice(len(fps_VAL_set), size = len(fps_VAL_set), replace=False)
    Indices_TR = np.random.choice(len(fps_TR_set), size = len(fps_TR_set), replace=False)

    fps_VAL_set = np.asarray(fps_VAL_set)
    fps_TR_set = np.asarray(fps_TR_set)
    L_VAL = np.asarray(L_VAL)
    L_TR = np.asarray(L_TR)
    Lb_VAL = np.asarray(Lb_VAL)
    Lb_TR = np.asarray(Lb_TR)

    fps_VAL_set = fps_VAL_set[Indices_VAL]
    fps_TR_set = fps_TR_set[Indices_TR]
    L_VAL = L_VAL[Indices_VAL]
    L_TR = L_TR[Indices_TR]
    Lb_VAL = Lb_VAL[Indices_VAL]
    Lb_TR = Lb_TR[Indices_TR]

    for path in fps_TR_set:
        image = nib.load(path)
        img = image.get_fdata()
        img = (img-img.min())/img.max()
        TR_set.append(img)

    for path in fps_VAL_set:
        image = nib.load(path)
        img = image.get_fdata()
        img = (img-img.min())/img.max()
        VAL_set.append(img)
        
    TR_set = np.asarray(TR_set)
    VAL_set = np.asarray(VAL_set)
    TR_set = np.reshape(TR_set,(len(TR_set),TR_set[0].shape[0],TR_set[0].shape[1],TR_set[0].shape[2],1))
    VAL_set = np.reshape(VAL_set,(len(VAL_set),VAL_set[0].shape[0],VAL_set[0].shape[1],VAL_set[0].shape[2],1))
    L_TR = np.asarray(L_TR)
    L_VAL = np.asarray(L_VAL)
    Lb_TR = np.asarray(Lb_TR)
    Lb_VAL = np.asarray(Lb_VAL)
    
    return TR_set,VAL_set,L_TR,L_VAL,Lb_TR,Lb_VAL

def custom_loss(y_true, y_pred):
    #return np.dot(K.categorical_crossentropy(y_true[:,:2], y_pred[:,:2]),y_pred[:,2:])
    #return np.dot(K.binary_crossentropy(y_true[:,:2], y_pred[:,:2]),y_pred[:,2:])
    return (K.categorical_crossentropy(y_true[:,:2], y_pred[:,:2]))*y_pred[:,-1]

def get_Merged_model(summary=False):
    """ Return the Keras model of the network
    """
    Sequential_model = get_model()
    image_input = Input(shape=(110, 110, 110, 1,))
    Sequential_Output = Sequential_model(image_input)
    input_for_loss = Input(shape=(1,))

    Merged_Output = keras.layers.concatenate([Sequential_Output, input_for_loss],axis=1)

    model = Model([image_input,input_for_loss], Merged_Output)

    if summary:
        print(model.summary())
    return model

def custom_acc(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:,:2], y_pred[:,:2])
    #return keras.metrics.categorical_accuracy(y_true, y_pred)

def pre_train(TR_set,VAL_set,L_TR,L_VAL,Lb_TR,Lb_VAL):
    #model = get_model(summary=True)
    model = get_model()
 
    opt = keras.optimizers.Adam(lr)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    history_train = model.fit(TR_set, L_TR, epochs = n_epochs , batch_size = 5, verbose = 0, shuffle = True, validation_data = (VAL_set,L_VAL))

    with open('history_pre_train_MFC',"a+") as f:
            for i in range(n_epochs):
                f.write('{},{},{},{},{}\n'.format(str(i), str(history_train.history['loss'][i]), str(history_train.history['acc'][i]),str(history_train.history['val_loss'][i]), str(history_train.history['val_acc'][i])))

    L_pred = model.predict(VAL_set)
    #print("@@@@@@@@@@ L_pred : \n",L_pred)

    y_pred = model.predict(TR_set)

    weighs_TR = abs(np.round(y_pred)-L_TR)*abs(L_TR-Lb_TR)
    weighs_VAL = abs(np.round(L_pred)-L_VAL)*abs(L_VAL-Lb_VAL)

    #print("@@@@@@@@@@ weighs_TR: \n",weighs_TR)
    #print("@@@@@@@@@@ weighs_VAL: \n",weighs_VAL)

    np.save('weighs_TR',weighs_TR[:,0])
    np.save('weighs_VAL',weighs_VAL[:,0])

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(L_VAL[:,0], L_pred[:,0])
    auc_keras = auc(fpr_keras, tpr_keras)
    #print("@@@@@@@@@@ fpr_keras : ",fpr_keras,"@@@@@@@@@@ tpr_keras : ",tpr_keras)

    return auc_keras

def train(k,TR_set,VAL_set,L_TR,L_VAL):
    weighs_TR = np.load('weighs_TR.npy')*k
    weighs_VAL = np.load('weighs_VAL.npy')*k
    weighs_TR[weighs_TR==0] = 1
    weighs_VAL[weighs_VAL==0] = 1
    #print("weighs_TR : ",weighs_TR)
    #print("weighs_VAL : ",weighs_VAL)
    with open('history_weighs',"a+") as f:
        f.write('k:{},weighs_TR:{},weighs_VAL:{},\n'.format(k,weighs_TR,weighs_VAL))    

    #model = get_Merged_model(summary=True)
    model = get_Merged_model()
    opt = keras.optimizers.Adam(lr)
    model.compile(loss=custom_loss,optimizer=opt, metrics=[custom_acc])
    #model.compile(loss=custom_loss,optimizer=opt, metrics=['accuracy'])
    #print(L_TR.shape,weighs_TR.shape,(np.zeros((L_TR.shape[0],1))).shape,(np.zeros((weighs_TR.shape[0],1))).shape)
    L_TR = np.concatenate([L_TR,np.zeros((L_TR.shape[0],1))],axis=1)
    L_VAL = np.concatenate([L_VAL,np.zeros((L_VAL.shape[0],1))],axis=1)
    #L_TR = np.concatenate([L_TR,np.ones((L_TR.shape[0],L_TR.shape[1]))],axis=1)
    #L_VAL = np.concatenate([L_VAL,np.ones((L_VAL.shape[0],L_VAL.shape[1]))],axis=1)

    history_train = model.fit([TR_set,weighs_TR], L_TR, epochs = n_epochs , batch_size = 5, verbose = 0, shuffle = True, validation_data = ([VAL_set,weighs_VAL],L_VAL))
    with open('history_train_MFC_custom_loss',"a+") as f:
            for i in range(n_epochs):
                f.write('{},{},{},{},{}\n'.format(str(i), str(history_train.history['loss'][i]), str(history_train.history['custom_acc'][i]),str(history_train.history['val_loss'][i]), str(history_train.history['val_custom_acc'][i])))
    
    L_pred = model.predict([VAL_set,weighs_VAL])
    #print("@@@@@@@@@@ L_pred : \n",L_pred)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(L_VAL[:,0], L_pred[:,0])
    auc_keras = auc(fpr_keras, tpr_keras)
    return auc_keras

def main():
    for i in range(0, 50, 1):
        TR_set,VAL_set,L_TR,L_VAL,Lb_TR,Lb_VAL = get_dataset() 
        auc_keras = pre_train(TR_set,VAL_set,L_TR,L_VAL,Lb_TR,Lb_VAL)
        for k in range(8, 50, 8):
            auc_keras_new = train(k,TR_set,VAL_set,L_TR,L_VAL)
            #print(i,k,auc_keras,auc_keras_new)
            with open('AUC_history_MFC',"a+") as f:
                f.write('{},{},{},{}\n'.format(i,k,auc_keras,auc_keras_new))

main()