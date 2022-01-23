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

    # read file matrix and convert to array
    testData = np.array(pd.read_csv('test.csv', header=None))
    # read target file for scores
    #targetData = np.array(pd.read_csv('target.csv', header=None))

    # obtain feature matrix
    feat_mat = feat_extr_test(testData)
    #standardizer = StandardScaler()
    #feat_mat = standardizer.fit_transform(feat_mat)

    # load model from pickle save
    filename = 'model.pkl'
    load_model = pickle.load(open(filename, 'rb'))

    # get result of model
    result = load_model.predict(feat_mat)
    result = np.reshape(result, (-1,1)). astype(int)

    # save results to Result.csv
    #np.savetxt('Result.csv', result, delimiter=',')
    pd.DataFrame(result).to_csv("Result.csv", sep=',', header=False, index=False )

    print("Result.csv file created!")
    pass

def feat_extr_test(data_mat):
    # variable to store list of lists (the inner list is list of 8 params)
    feat_mat_list = []
    # for each row create a list of 8 values (8 feature paramters)
    for idx, row in enumerate(data_mat):
        # create a list of 8 values/paramaeters
        feat_list = []

        # for each row consider minimum value index as tm value
        tm = np.nanargmin(row)

        # PARAMETER 1: difference between time at CGMMax and time at meal taken
        tau = (np.nanargmax(row) - tm)*5
        feat_list.insert(len(feat_list), tau)

        # PARAMETER 2: difference between CGMMax and CGM at time of meal taken
        CGMDiffNorm = (np.nanmax(row) - row[tm])#/row[tm]
        feat_list.insert(len(feat_list), CGMDiffNorm)

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
        feat_list.insert(len(feat_list), abs(sec_max))
        feat_list.insert(len(feat_list), sec_max_ind)
        feat_list.insert(len(feat_list), abs(thrd_max))
        feat_list.insert(len(feat_list), thrd_max_ind)

        # PARAMETER 7: differential of CGM data (GET MAX VALUE)
        feat_list.insert(len(feat_list), np.nanmax(np.diff(row)))

        # PARAMETER 8: double differential of CGM data
        feat_list.insert(len(feat_list), np.nanmax(np.diff(row, n=2)))

        # PARAMETER 9: 

        # add updated feature to feat_mat_list
        feat_mat_list.append(feat_list)

    return np.array(feat_mat_list)

if __name__ == "__main__":
    main()