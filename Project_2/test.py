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
from train import feat_extr

def main():

    # read file matrix and convert to array
    testData = np.array(pd.read_csv('test.csv', header=None))
    # read target file for scores
    #targetData = np.array(pd.read_csv('target.csv', header=None))

    # obtain feature matrix
    feat_mat = feat_extr(testData)
    #standardizer = StandardScaler()
    #feat_mat = standardizer.fit_transform(feat_mat)

    # load model from pickle save
    filename = 'mod.pkl'
    load_model = pickle.load(open(filename, 'rb'))

    # get result of model
    result = load_model.predict(feat_mat)
    result = np.reshape(result, (-1,1))

    # save results to Result.csv
    #np.savetxt('Result.csv', result, delimiter=',')
    pd.DataFrame(result).to_csv("Result.csv", sep=',', header=False, index=False )

    print("Result.csv file created!")
    pass

if __name__ == "__main__":
    main()