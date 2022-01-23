# RUN COMMAND: python main.py ./Project-1-Files/CGMData.csv ./Project-1-Files/InsulinData.csv

# main.py
import sys
import numpy as np
import pandas as pd
import sklearn
from scipy.fft import fft, ifft
#from sklearn.model_selection import train_test_split
#----------------------PREPROCESSING-------------------------------
from sklearn.preprocessing import StandardScaler
#--------------------CLASSIFIERS TRIED------------------------------
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#--------------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pickle

def main():

    # assume CGMData.csv, CGM_patient2.csv and InsulinData.csv, Insulin_patient2.csv files are in compilation and execution folder
    # read data from the files - parse Date and Time columns as DateTime
    cgmDataPat_1_orig = pd.read_csv('CGMData.csv', parse_dates=[['Date', 'Time']])
    cgmDataPat_2_orig = pd.read_csv('CGM_patient2.csv', parse_dates=[['Date', 'Time']])
    insDataPat_1_orig = pd.read_csv('InsulinData.csv', parse_dates=[['Date', 'Time']])
    insDataPat_2_orig = pd.read_csv('Insulin_patient2.csv', parse_dates=[['Date', 'Time']])

    # STORE ORIG DATA IN VARIABLES (to be able to make changes and avoid reading csv everytime)
    # focus only on certain columns relevant to the dataset and Project 2
    cgmDataPat_1 = cgmDataPat_1_orig.loc[:, ['Date_Time', 'Sensor Glucose (mg/dL)']]
    cgmDataPat_2 = cgmDataPat_2_orig.loc[:, ['Date_Time', 'Sensor Glucose (mg/dL)']]
    insDataPat_1 = insDataPat_1_orig.loc[:, ['Date_Time', 'BWZ Carb Input (grams)']]
    insDataPat_2 = insDataPat_2_orig.loc[:, ['Date_Time', 'BWZ Carb Input (grams)']]

    #-------------------------------------------  MEAL DATA
    # run the extract_MD_DT function to get valid Date_Time values
    insValDT_1, insCarbDT_1 = extractMDDT(insDataPat_1)
    insValDT_2, insCarbDT_2 = extractMDDT(insDataPat_2)

    # SET THRESHOLD FOR NaN/MISSING/0 VALUES (in percentage)
    threshold_MD = 50

    # get sensor values in cgmData for valid DTs
    mealDictPat_1 = getMDFeatMat(insValDT_1, cgmDataPat_1, threshold_MD)
    mealDictPat_2 = getMDFeatMat(insValDT_2, cgmDataPat_2, threshold_MD)

    # thus we get two matrices for two patients for MEAL DATA
    mat_MDPat_1 = mealDictPat_1['ft_mat']
    mat_MDPat_2 = mealDictPat_2['ft_mat']

    #----------------------------------------   NO MEAL DATA
    # valid Meal DateTimes that are present in the feature matrix
    # building on that we decide no-meal times
    cgmValDT_1 = mealDictPat_1['cgmValDT']
    cgmValDT_2 = mealDictPat_2['cgmValDT']

    threshold_NMD = 60

    mat_NMDPat_1 = getNMDFeatMat(insCarbDT_1, cgmValDT_1, cgmDataPat_1, threshold_NMD)
    mat_NMDPat_2 = getNMDFeatMat(insCarbDT_2, cgmValDT_2, cgmDataPat_2, threshold_NMD)

    #print("Matrices shape for each patient for MD")
    #print("Pat_1_MD:"+str(mat_MDPat_1.shape))
    #print("Pat_2_MD:"+str(mat_MDPat_2.shape))

    #print("Matrices shape for each patient for NMD")
    #print("Pat_1_NMD:"+str(mat_NMDPat_1.shape))
    #print("Pat_2_NMD:"+str(mat_NMDPat_2.shape))

    #-------------------------------------    FEATURE EXTRACTOR

    # merge all rows under md and nmd separately
    mat_MD = np.row_stack( (mat_MDPat_1, mat_MDPat_2) )
    mat_NMD = np.row_stack( (mat_NMDPat_1, mat_NMDPat_2) )

    # standardize
    #standardizer_b4_ft = StandardScaler()
    #mat_MD = standardizer_b4_ft.fit_transform(mat_MD)
    #mat_NMD = standardizer_b4_ft.fit_transform(mat_NMD)

    md_ft = feat_extr(mat_MD)
    nmd_ft = feat_extr(mat_NMD)

    #print("FEAT Matrices shape for each patient for MD")
    #print("Pat_1_MD_Feat:"+str(md_ft_1.shape))
    #print("Pat_2_MD_Feat:"+str(md_ft_2.shape))

    #print("FEAT Matrices shape for each patient for NMD")
    #print("Pat_1_NMD_Feat:"+str(nmd_ft_1.shape))
    #print("Pat_2_NMD_Feat:"+str(nmd_ft_2.shape))

    # combine the matrices and add labels to make (P+Q)x(8+1) DATA MATRIX
    # shuffle each md and nmd data
    #np.random.shuffle(md_ft)
    #np.random.shuffle(nmd_ft)
    # make them equal size of data for both labels
    #equal_size = int(min(md_ft.shape[0], nmd_ft.shape[0]))
    #nmd_size = int(equal_size)
    #if nmd_size >= nmd_ft.shape[0]:
        #nmd_size = nmd_ft.shape[0]
    #md_ft = md_ft[:equal_size,:]
    #nmd_ft = nmd_ft[:nmd_size,:]

    # merge all rows again for feature matrix (w/o labels)
    feature_mat = np.row_stack( (md_ft, nmd_ft) )

    # add class labels of 1 for md and 0 for nmd
    class_lbl_ones = np.ones( md_ft.shape[0] )
    class_lbl_zeros = np.zeros( nmd_ft.shape[0] )

    # total class labels
    class_lbls = np.row_stack( ( np.reshape(class_lbl_ones, (-1,1)), np.reshape(class_lbl_zeros, (-1, 1)) ) )

    # add a single column of class labels to the md and nmd matrix
    md_ft_lbls = np.column_stack( (md_ft, class_lbl_ones) )
    nmd_ft_lbls = np.column_stack( (nmd_ft, class_lbl_zeros) )

    # combine for FULL DATA MATRIX
    data_mat = np.column_stack( (feature_mat, class_lbls) )
    # shuffle final data too
    #np.random.shuffle(data_mat)

    #print("MD Feat Labels shape:"+str(md_ft_lbls.shape))
    #print("NMD Feat Labels shape:"+str(nmd_ft_lbls.shape))
    #print("")
    #print("DATA MATRIX shape:"+str(data_mat.shape))

    #-------------------------------------    TRAINING SPLIT
    # take 80% of meal data as train split
    md_len = md_ft_lbls.shape[0]
    md_len_80 = int(0.8*md_len)
    tr_md_ft_lbls = md_ft_lbls[:md_len_80, :]
    ts_md_ft_lbls = md_ft_lbls[:(md_len-md_len_80), :-1]
    tg_md_ft_lbls = md_ft_lbls[:(md_len-md_len_80), -1]

    # take 80% of no meal data as train split
    nmd_len = nmd_ft_lbls.shape[0]
    nmd_len_80 = int(0.8*nmd_len)
    tr_nmd_ft_lbls = nmd_ft_lbls[:nmd_len_80, :]
    ts_nmd_ft_lbls = nmd_ft_lbls[:(nmd_len-nmd_len_80), :-1]
    tg_nmd_ft_lbls = nmd_ft_lbls[:(nmd_len-nmd_len_80), -1]

    # combine md and nmd train data
    X_tr = np.row_stack( ( tr_md_ft_lbls[:, :-1], tr_nmd_ft_lbls[:, :-1] ) )
    y_tr = np.row_stack( ( np.reshape(tr_md_ft_lbls[:, -1], (-1,1)), np.reshape(tr_nmd_ft_lbls[:, -1], (-1,1)) ) )
    # combine md and nmd test data
    X_ts = np.row_stack( ( ts_md_ft_lbls, ts_nmd_ft_lbls ) )
    y_ts = np.row_stack( ( np.reshape(tg_md_ft_lbls, (-1,1)), np.reshape(tg_nmd_ft_lbls, (-1,1)) ) )

    # save test features and target variables to csv
    #pd.DataFrame(X_ts).to_csv("my_test.csv", sep=',', header=False, index=False )
    #pd.DataFrame(y_ts).to_csv("target.csv", sep=',', header=False, index=False )

    # create another set of variables to train on entire dataset
    X_tr_full = data_mat[:,:-1]#np.row_stack( (X_tr, X_ts) )
    y_tr_full = data_mat[:,-1:]#np.row_stack( ( np.reshape(y_tr, (-1,1)), np.reshape(y_ts, (-1,1)) ) )

    #print("X_Train shape:"+str(X_tr.shape))
    #print("y_Train shape:"+str(y_tr.shape))
    #print("X_Test shape:"+str(X_ts.shape))
    #print("y_Test shape:"+str(y_ts.shape))

    #-------------------------------------    MODEL TRAINING (SVM)
    #standardizer = StandardScaler()
    #X_tr_full = standardizer.fit_transform(X_tr_full)

    model = RandomForestClassifier(criterion="entropy", max_depth=5, n_estimators=10, max_features=1)#MLPClassifier(alpha=0.001, hidden_layer_sizes=(3,3,2,2,))
    model.fit(X_tr_full, y_tr_full.ravel())
    # save the fitted model to pickle
    filename = 'mod.pkl'
    pickle.dump(model, open(filename, 'wb'))

    print("LEARNED MODEL SAVED IN model1.pkl!")

    #print("CROSS VALIDATING:-")
    #splits = cross_val_splits(md_ft_lbls, nmd_ft_lbls, 5)

    #print("SCORES:-")
    #cross_val_scores(splits, model)

    pass

# define functions for getting train data
# MD - Meal Data and DT - Date_Time
# passes the insulinData to find valid meal DateTimes
def extractMDDT(ins_extrct_DT):
    # create condition to check non-NaN, non-Zero carb input values
    carbInputCond = ~ins_extrct_DT['BWZ Carb Input (grams)'].isna() & ins_extrct_DT['BWZ Carb Input (grams)']!=0
    # select those Date_Times
    insCarbDT = ins_extrct_DT[carbInputCond].copy(deep=True)
    
    # the latest date_time value is considered MEAL DATA
    # later we can check if the amount of data in this period is <80% then remove
    # else we keep it
    DT_max_2h1m = max(insCarbDT['Date_Time']) + pd.DateOffset(hours=2, minutes=1)
    
    # create columns for next Date_TIME(DT) and DT+2hrs
    insCarbDT['Next_DT'] = insCarbDT['Date_Time'].shift(1, fill_value=DT_max_2h1m)
    insCarbDT['DT+2hrs'] = insCarbDT['Date_Time'] + pd.DateOffset(hours=2)
    
    # conddition to check if Next_DT is between DT and DT+2hrs
    btw2HrsCond = insCarbDT['Next_DT'].between(insCarbDT['Date_Time'], insCarbDT['DT+2hrs'], inclusive='neither')
    insCond2DT = insCarbDT.loc[~btw2HrsCond]
    
    # because we set the between function to have inclusive = 'neither' it keeps cond 3 true
    # Now, only extract the Date_Time values
    insValDT = insCond2DT.loc[:, ['Date_Time']]
    
    return insValDT, insCarbDT

def getMDFeatMat(insValDT_, cgmDataPat_, threshold):
    # for each of valid Date_time get corresponding CGM DT after (but earliest) time
    cgmValDT = insValDT_['Date_Time'].apply( lambda x: cgmDataPat_[cgmDataPat_['Date_Time'] >= x]['Date_Time'].min() )
    
    cgmDataList = []
    lengthsList = []
    
    for (ind, val) in cgmValDT.iloc[::-1].iteritems():
        # different masks give different possible lengths of sensor values
        # the chosen mask exclude the time exactly at tm+2hrs (gives more samples to train)
        mask = (cgmDataPat_['Date_Time'] >= val-pd.DateOffset(minutes=30)) & (cgmDataPat_['Date_Time'] < val+pd.DateOffset(hours=2))
        sens_vals = np.array(cgmDataPat_.loc[mask]['Sensor Glucose (mg/dL)'].iloc[::-1])
        
        nonNALen = np.count_nonzero(~np.isnan(sens_vals))
        
        # we only want the data if the non-NaN is 80% or more of 30 (>=24)
        # AND also the length of data is 30
        if (len(sens_vals)==30 and nonNALen >= (threshold/100)*30):
            cgmDataList.append( sens_vals )
            lengthsList.append( nonNALen )
            
    cgmDataList = np.vstack(cgmDataList)
    return {'ft_mat': cgmDataList, 'lengths': lengthsList, 'cgmValDT':cgmValDT}

# function to extract No Meal Data feature matrix
# NMD for No Meal Data
def getNMDFeatMat(insVal_, cgmMDVal_, cgmDataPat_, threshold):
    # from the insVal_ get datetime in cgm table
    cgmVal_ = insVal_['Date_Time'].apply( lambda x: cgmDataPat_[cgmDataPat_['Date_Time'] >= x]['Date_Time'].min() )
    
    # get the max DT for the cgmData of each patient
    cgmMaxDT = cgmDataPat_['Date_Time'].max()
    
    # reverse the series values (chrono order)
    cgmVal_ = cgmVal_[::-1]
    
    # add maximum DT to the series
    cgmVal_ = cgmVal_.append(pd.Series(cgmMaxDT), ignore_index=True)
    
    # variable storing list of lists of CGM values of No Meal Data
    cgmNMDataList = []
    
    # loop through the valid Meal DTs in cgmVal
    # start from very beginning of dataset
    for ind, val in cgmVal_[:-1].iteritems():
        # MDT = Meal DateTime
        cur_MDT = val
        #print("cur_MDT: "+str(cur_MDT))
        cur_MDT_2h = cur_MDT + pd.DateOffset(hours=2)
        #print("cur_MDT_2h: "+str(cur_MDT_2h))
        next_MDT = cgmVal_[ind+1]
        #print("next_MDT: "+str(next_MDT))
        # loop to get multiple 2hrs intervals
        # start loop at cur_MDT_2h to cur_MDT_2h + 2hrs
        start_MDT = cur_MDT_2h
        #print("start_MDT: "+str(start_MDT))
        next_2hr_MDT = cur_MDT_2h + pd.DateOffset(hours=2)
        #print("next_2hr_MDT: "+str(next_2hr_MDT))

        # we want to look for 2hour intervals until next_MDT
        # if next_MDT is a value in cgmMDVal_ then it was a valid meal DT
        # only for such DTs we use next_MDT_lim which is 30 min before the next_MDT
        next_MDT_off = next_MDT - pd.DateOffset(minutes=30)
        #if next_MDT not in cgmMDVal_:
            #next_MDT_off = next_MDT
        while(next_2hr_MDT <= next_MDT_off):            
            mask = ((cgmDataPat_['Date_Time'] > start_MDT) & (cgmDataPat_['Date_Time'] <= next_2hr_MDT))
            # get values of that mask added to cgmNMDataList
            mask_vals = np.array(cgmDataPat_.loc[mask]['Sensor Glucose (mg/dL)'])
            nonNALen = np.count_nonzero(~np.isnan(mask_vals))
            if (len(mask_vals)==24 and nonNALen >= (threshold/100)*24):
                cgmNMDataList.append(cgmDataPat_.loc[mask]['Sensor Glucose (mg/dL)'][::-1].tolist())
            # update vaues
            start_MDT = next_2hr_MDT
            #print("start_MDT: "+str(start_MDT))
            next_2hr_MDT = next_2hr_MDT + pd.DateOffset(hours=2)
            #print("next_2hr_MDT: "+str(next_2hr_MDT))
        
    # return the feature matrix
    return np.array(cgmNMDataList)

def feat_extr(data_mat):
    # variable to store list of lists (the inner list is list of 8 params)
    feat_mat_list = []
    # for each row create a list of 8 values (8 feature paramters)
    for idx, row in enumerate(data_mat):
        # create a list of 8 values/paramaeters
        feat_list = []

        # for each row consider minimum value index as tm value
        tm = np.nanargmin(row)

        # PARAMETER 1: difference between time at CGMMax and time at meal taken
        tau = (np.nanargmax(row) - tm)
        #feat_list.insert(len(feat_list), tau)

        # PARAMETER 2: difference between CGMMax and CGM at time of meal taken
        CGMDiffNorm = (np.nanmax(row) - row[tm])#/row[tm]
        #feat_list.insert(len(feat_list), CGMDiffNorm)

        # PARAMETER 3,4,5,6: get FFT magnitude and frequency bins
        nonNARow = row[~np.isnan(row)]
        # fft
        fft_vals = fft( nonNARow )
        # extract second and third peaks
        fft_vals_copy = fft_vals
        fft_vals_copy.sort()
        sec_max = fft_vals_copy[-2]
        thrd_max = fft_vals_copy[-3]
        # extract the indices of second and third peaks
        sec_max_ind = np.where(fft_vals==sec_max)[0][0]
        thrd_max_ind = np.where(fft_vals==thrd_max)[0][0]
        # add values to feat list
        #feat_list.insert(len(feat_list), abs(sec_max))
        #feat_list.insert(len(feat_list), sec_max_ind)
        #feat_list.insert(len(feat_list), abs(thrd_max))
        #feat_list.insert(len(feat_list), thrd_max_ind)

        # PARAMETER 7: differential of CGM data (GET MAX VALUE)
        #feat_list.insert(len(feat_list), np.nanmax(np.diff(row)))

        # PARAMETER 8: double differential of CGM data
        feat_list.insert(len(feat_list), np.nanmax(np.diff(row, n=2)))

        # PARAMETER 9: 

        # add updated feature to feat_mat_list
        feat_mat_list.append(feat_list)

    return np.array(feat_mat_list)

# 80% train and 20% test means 5 splits
def cross_val_splits(md_data, nmd_data, num_split):
    # print total data length
    #print("TOTAL DATA LEN = "+str(len(md_data)+len(nmd_data)))
    
    # make num_split
    splits = [None]*num_split
    
    # fraction of test data
    frac_test = 1/num_split
    
    # split md_data
    md_len = md_data.shape[0]
    md_frac_test_len = int(frac_test*md_len)
    #print("MD Frac Test: "+str(md_frac_test_len))
    # slpit no meal data
    nmd_len = nmd_data.shape[0]
    nmd_frac_test_len = int(frac_test*nmd_len)
    #print("NMD Frac Test: "+str(nmd_frac_test_len))
    
    # start test split at 0th index
    md_test_split_start = 0
    nmd_test_split_start = 0
    for ind in range(num_split):
        # this is the ind-th split
        # end of split index at
        md_test_split_end = md_test_split_start + md_frac_test_len
        nmd_test_split_end = nmd_test_split_start + nmd_frac_test_len
        #print("PRINTING INDICES: ........")
        #print("Split "+str(ind)+"-MD Start-End: "+str(md_test_split_start)+" - "+str(md_test_split_end))
        #print("Split "+str(ind)+"-NMD Start-End: "+str(nmd_test_split_start)+" - "+str(nmd_test_split_end))
        # create a dictionary of Train(MD+NMD), Test(MD+NMD)
        # Test part first
        test_md = md_data[ md_test_split_start:md_test_split_end, : ]
        test_nmd = nmd_data[ nmd_test_split_start:nmd_test_split_end, : ]
        # Train part - will have two parts (BEFORE and AFTER test section)
        train_md_b4 = md_data[ :md_test_split_start, : ]
        train_md_af = md_data[ md_test_split_end:, : ]
        train_nmd_b4 = nmd_data[ :nmd_test_split_start, : ]
        train_nmd_af = nmd_data[ nmd_test_split_end:, : ]
        
        train_md = np.row_stack( (train_md_b4, train_md_af) )
        train_nmd = np.row_stack( (train_nmd_b4, train_nmd_af) )
        # combine MD and NMD
        train_comb = np.row_stack( (train_md,train_nmd) )
        test_comb = np.row_stack( (test_md,test_nmd) )
        splits[ind] = {
            'train' : train_comb,
            'test' : test_comb
        }
        
        # update the split indices
        md_test_split_start = md_test_split_end
        nmd_test_split_start = nmd_test_split_end
        # print shapes
        #print("PRINTING SHAPES: ........")
        #print("Split "+str(ind)+"-Train Shape: "+str(train_comb.shape))
        #print("Split "+str(ind)+"-Test Shape: "+str(test_comb.shape))
        
    return splits

# function to run model and get cross validation scores
# function to run model and get cross validation scores
def cross_val_scores(splits, model):
    # 4 variables
    accu_list = [None]*len(splits) # accuracy
    # for label 1
    p_list_1 = [None]*len(splits)    # precision
    r_list_1 = [None]*len(splits)    # recall
    f_list_1 = [None]*len(splits)    # F1 score
    # for label 0
    p_list_0 = [None]*len(splits)    # precision
    r_list_0 = [None]*len(splits)    # recall
    f_list_0 = [None]*len(splits)    # F1 score
    
    for ind in range(len(splits)):
        split_tr = splits[ind]['train']
        split_ts = splits[ind]['test']
        
        # fit to train
        model.fit(split_tr[:,:-1], split_tr[:,-1].ravel())
        # test
        res = model.predict(split_ts[:,:-1])
        # ground
        grd = split_ts[:,-1].ravel()
        # scores
        accu = accuracy_score(grd, res)
        accu_list[ind] = accu
        #print("Accuracy: "+str(accu))
        scores_1 = precision_recall_fscore_support(grd, res, average='binary', pos_label=1)
        scores_0 = precision_recall_fscore_support(grd, res, average='binary', pos_label=0)
        p_list_1[ind] = scores_1[0]
        p_list_0[ind] = scores_0[0]
        #print("Precision: "+str(scores[0]))
        r_list_1[ind] = scores_1[1]
        r_list_0[ind] = scores_0[1]
        #print("Recall: "+str(scores[1]))
        f_list_1[ind] = scores_1[2]
        f_list_0[ind] = scores_0[2]
        #print("F1: "+str(scores[2]))
        
    print("Accuracy: "+str(np.mean(accu_list)))
    print("Precision for 1: "+str(np.mean(p_list_1)))
    print("Recall for 1: "+str(np.mean(r_list_1)))
    print("F1 for 1: "+str(np.mean(f_list_1)))
    print("----------------")
    print("Precision for 0: "+str(np.mean(p_list_0)))
    print("Recall for 0: "+str(np.mean(r_list_0)))
    print("F1 for 0: "+str(np.mean(f_list_0)))

if __name__ == "__main__":
    main()