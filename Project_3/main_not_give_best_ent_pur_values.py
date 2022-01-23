# RUN COMMAND: python main.py

# main.py
import sys
import numpy as np
import pandas as pd
import sklearn
import pickle
from scipy.fft import fft, ifft
# IMPORT REQUIRED CLUSTERING MODULES
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

def main():

    # PARAMS SET
    # set threshold for NaN in meal data
    threshold = 70
    # select which features to target out of 8
    slct_ft = [1, 1, 1, 1, 1, 1, 1, 1]
    # set k-means centroid initialization method
    # and number of iterations for a good Entropy and Purity
    km_init = "random"
    km_iters = 100
    # set DBSCAN params
    eps_set = 213
    mpts_set = 326    

    # read data from the files - parse Date and Time columns as DateTime
    cgmDataOrig = pd.read_csv('CGMData.csv', parse_dates=[['Date', 'Time']])
    insDataOrig = pd.read_csv('InsulinData.csv', parse_dates=[['Date', 'Time']])

    # STORE ORIG DATA IN VARIABLES (to be able to make changes and avoid reading csv everytime)
    # focus only on certain columns relevant to the dataset and Project 2
    # focus only on certain columns relevant to the dataset and Project 3
    cgmData = cgmDataOrig.loc[:, ['Date_Time', 'Sensor Glucose (mg/dL)']]
    insData = insDataOrig.loc[:, ['Date_Time', 'BWZ Carb Input (grams)']]

    # ------------------------------- MEAL DATA PREP
    print("PREPING MEAL DATA ...")
    # run the validInsulinDTCarbs on insulin data
    insDT_Carbs = validInsulinDTCarbs(insData)
    
    mdMat_Carbs, lenList = sensValAndCarbInput(insDT_Carbs, cgmData, threshold)
    # NOTE: last column in matrix is Carbs Input for each Meal Data and comes from Insulin Data

    # pass all but final (Carbs) column to feature extractor
    mdFT = feat_extr(mdMat_Carbs[:,:-1], slct_ft)
    # variable storing only carbs values
    mdCarbs = mdMat_Carbs[:,-1].reshape((-1,1))
    # rejoin the carbs column to the extracted features
    mdFT_Carbs = np.column_stack( (mdFT, mdMat_Carbs[:,-1]) )
    print("FEAT Matrix shape for MD"+str(mdFT.shape))
    print("CARBS Matrix shape for MD"+str(mdCarbs.shape))

    # ----------------------------- BINS AND GROUND TRUTH
    print("COMPUTING GROUND TRUTHS ...")
    # ravel the 1 column matrix to 1D array
    mdCarbs_rav = mdCarbs.ravel()

    # get max and min of the Carbs values
    max_Carbs = mdCarbs_rav.max()
    min_Carbs = mdCarbs_rav.min()

    # REQUIRED number of clusters are
    req_num_clusters = int(np.rint((max_Carbs - min_Carbs)/20))

    bins = np.arange(min_Carbs+20, max_Carbs, 20)
    # make sure the min value is included in the first bin
    #bins[0] = min_Carbs-1

    # get the ground truth values as bins
    mdGT = np.digitize(mdCarbs_rav, bins, right=True)
    # SUBTRACTED 1 to get bin values starting at 0

    # join the feature matrix (FT) and ground truth (GT) values
    mdFT_GT = np.column_stack( (mdFT, mdGT) )

    print("GRND TRUTH values unique count = "+str( np.unique(mdGT, return_counts=True) ))

    print("GRND TRUTH shape - FT_GT shape = "+str(mdFT_GT.shape))

    # -----------------------------ANY STANDARDIZE/NORMALIZE
    # normalize data
    #norm_mdMat= normalize(mdMat_Carbs[:,:-1], axis=1, norm='l2')
    #mdFT = norm_mdFT

    # standardize data
    #scaler = MinMaxScaler()
    #mdFT = scaler.fit_transform(mdFT)

    # ----------------------------- CLUSTERING DATA - KMEANS
    print("CLUSTERING WITH K-MEANS ...")
    # create k-means model
    mod_KM = KMeans(n_clusters = req_num_clusters, init=km_init)
    fitted_KM = mod_KM.fit( mdFT )

    # RAISE ERROR: if the max iterations is reached
    assert fitted_KM.n_iter_ < fitted_KM.max_iter, "KMEANS ERROR: took more than max_iters (no convergence)!"

    # the labels given by KMeans
    labels_KM = fitted_KM.labels_

    # create list of clusters with the data points
    lbls_uvals_KM = np.unique(labels_KM)
    clust_list_KM = [ mdFT_GT[np.where(labels_KM==lbl)] for lbl in lbls_uvals_KM ]

    # NUMBER OF POINTS IN EACH LABELS
    print("NUM POINTS for each label (KM) :- "+str( np.unique(labels_KM, return_counts=True) ))
    print("NUM POINTS in each cluster (KM) :- "+str( [ len(clst) for clst in clust_list_KM ] ))

    # ----------------------------- CLUSTERING DATA - DBSCAN
    print("CLUSTERING WITH DBSCAN ...")
    # create DBSCAN model
    mod_DB = DBSCAN(eps=eps_set, min_samples=mpts_set)
    fitted_DB = mod_DB.fit( mdFT )

    # the labels given by DBSCAN
    labels_DB = fitted_DB.labels_

    # NUMBER OF NOISE POINTS
    print("% of Noise Points = " + str( (sum(labels_DB == -1)/len(labels_DB))*100 ))

    # the number of clusters formed
    max_label_DB = np.max(labels_DB)
    # if the max number is -1 then RAISE ERROR
    assert max_label_DB != -1, "DBSCAN ERROR: all labels are -1!"
    # min label
    min_label_DB = int(np.nanmin( np.where(labels_DB<0, np.nan, labels_DB) ))

    print("Number of Clusters = " + str(max_label_DB - min_label_DB + 1))

    print("NUM POINTS in each label (DB):- "+str( np.unique(labels_DB, return_counts=True) ))

    # remove noise points 
    # no noise points (NoiseLess) condition
    cond_NL = np.where(labels_DB!=-1)
    labels_DB_NL = labels_DB[cond_NL]
    mdFT_NL = mdFT[cond_NL]
    mdGT_NL = mdGT[cond_NL]
    mdFT_GT_NL = mdFT_GT[cond_NL]

    # Bisecting K-Means until req_num_clusters is reached - PASS NOISELESS DATA
    clust_list_DB_BKM = bis_KM_till_k(mdFT_GT_NL, labels_DB_NL, req_num_clusters)

    print("NUM POINTS in each cluster (DB+BKM):- "+str( [ len(clst) for clst in clust_list_DB_BKM ] ))

    # ---------------------------- CLUSTER COMPARISON MATRIX
    print("COMPUTING CLUSTER COMPARISON MATRIX ...")
    # get the clust count comparison with the GT for KM
    comp_Mat_KM = clustCompConv(clust_list_KM, req_num_clusters)
    # and for DBKM clusters
    comp_Mat_DBKM = clustCompConv(clust_list_DB_BKM, req_num_clusters)

    # ---------------------------- COMPUTE SSE, ENTROPY, PURITY
    # SSE for K-Means
    clust_sse_KM = [ sseFunc(cl_) for cl_ in clust_list_KM ]
    sse_KM = sum(clust_sse_KM)
    print("SSE for KMeans = "+str(sse_KM))

    # SSE for DB_BKM
    clust_sse_DBKM = [ sseFunc(cl_) for cl_ in clust_list_DB_BKM ]
    sse_DBKM = sum(clust_sse_DBKM)
    print("SSE for DB+Kmeans = "+str(sse_DBKM))

    # ENTROPY for K-Means
    ent_KM = entropyOfClusters(comp_Mat_KM)
    print("ENTROPY for KMeans = "+str(ent_KM))

    # ENTROPY for DB_BKM
    ent_DBKM = entropyOfClusters(comp_Mat_DBKM)
    print("ENTROPY for DB-BKM = "+str(ent_DBKM))

    # PURITY for K-Means
    pur_KM = purityOfClusters(comp_Mat_KM)
    print("PURITY for KMeans = "+str(pur_KM))

    # PURITY for K-Means
    pur_DBKM = purityOfClusters(comp_Mat_DBKM)
    print("PURITY for DB-BKM = "+str(pur_DBKM))

    # ---------------------------- WRITE TO Result.csv
    print("WRITING FINAL VALUES TO csv FILE ...")
    # create a 1x6 matrix of result values
    res_mat = np.array([sse_KM, sse_DBKM, ent_KM, ent_DBKM, pur_KM, pur_DBKM]).reshape((1,-1))
    pd.DataFrame(res_mat).to_csv("Result.csv", sep=',', header=False, index=False )
    print("Result.csv created!")
    pass

# HELPER FUNCTIONS
# get the Valid MD DT and Carb values from Insulin data
def validInsulinDTCarbs(ins_data):
    carbInsCond = ~ins_data['BWZ Carb Input (grams)'].isna() & ins_data['BWZ Carb Input (grams)']!=0
    ins_data_carb = ins_data[carbInsCond].copy(deep=True)

    # the last date_time value is definitely MEAL DATA because it has its own 2hours of uninterrupted data
    # get 2 hours 1 min value over last date_time
    DT_max_2h1m = max(ins_data_carb['Date_Time']) + pd.DateOffset(hours=2, minutes=1)
    # create columns for next Date_TIME(DT) and DT+2hrs
    ins_data_carb['Next_DT'] = ins_data_carb['Date_Time'].shift(1, fill_value=DT_max_2h1m)
    ins_data_carb['DT+2hrs'] = ins_data_carb['Date_Time'] + pd.DateOffset(hours=2)

    # condition to check if Next_DT is between DT and DT+2hrs
    btw2HrsCond = ins_data_carb['Next_DT'].between(ins_data_carb['Date_Time'], ins_data_carb['DT+2hrs'], inclusive='neither')
    ins_data_carb = ins_data_carb.loc[~btw2HrsCond]

    # Now, only extract the Date_Time and Carb Input values
    insValDTCarbs = ins_data_carb.loc[:, ['Date_Time', 'BWZ Carb Input (grams)']]

    return insValDTCarbs

# get the CGM Sensor Values for each meal Data and the corresponding Carb Input value
def sensValAndCarbInput(insDT_Carbs_, cgmData_, threshold):
    # for each of valid Insulin Date_time get corresponding CGM DT after (but earliest) time
    cgmDT_Carbs_ = insDT_Carbs_['Date_Time'].apply(lambda x: cgmData_[cgmData_['Date_Time'] >= x]['Date_Time'].min())

    cgmDataList = []
    lengthsList = []

    # reverse the CGM DT Carbs values to make it chronologic order
    # while doing so loop through it
    for (ind, val) in cgmDT_Carbs_.iloc[::-1].iteritems():
        # for each value here we extract all Sensor Glucose values
        mask = (cgmData_['Date_Time'] >= val-pd.DateOffset(minutes=30)) & (cgmData_['Date_Time'] < val+pd.DateOffset(hours=2))
        sens_vals = np.array(cgmData_.loc[mask]['Sensor Glucose (mg/dL)'].iloc[::-1])
        nonNALen = np.count_nonzero(~np.isnan(sens_vals))
        # also add the Carb Input at the end
        carb_inp_val = insDT_Carbs_['BWZ Carb Input (grams)'][ind]
        comb_vals = np.append(sens_vals, carb_inp_val)

        # we only want the data if the non-NaN length is THRESHOLD% or more of 30 (>=24)
        # AND also the length of data must be 30
        if (len(sens_vals)==30 and nonNALen >= (threshold/100)*30):
            cgmDataList.append( comb_vals )
            lengthsList.append( nonNALen )

    cgmDataList = np.vstack(cgmDataList)

    # return list of lists as ndarray
    return np.array(cgmDataList), lengthsList

def feat_extr(data_mat, select_feat):
    # variable to store list of lists (the inner list is list of 8 params)
    feat_mat_list = []
    # for each row create a list of 8 values (8 feature paramters)
    for idx, row in enumerate(data_mat):
        # create a list of 8 values/paramaeters
        feat_list = []

        # for each row consider minimum value index as tm value
        tm = np.nanargmin(row)

        # PARAMETER 1: difference between time at CGMMax and time at meal taken
        if select_feat[0] == 1:
            tau = (np.nanargmax(row) - tm)*5
            feat_list.insert(len(feat_list), tau)

        # PARAMETER 2: difference between CGMMax and CGM at time of meal taken
        if select_feat[1] == 1:
            CGMDiffNorm = (np.nanmax(row) - row[tm])#/row[tm]
            feat_list.insert(len(feat_list), CGMDiffNorm)

        # PARAMETER 3,4,5,6: get FFT magnitude and frequency bins
        if ((select_feat[2] == 1) or (select_feat[3] == 1) or (select_feat[4] == 1) or (select_feat[5] == 1)):
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
            if select_feat[2] == 1:
                feat_list.insert(len(feat_list), abs(sec_max))
            if select_feat[3] == 1:
                feat_list.insert(len(feat_list), sec_max_ind)
            if select_feat[4] == 1:
                feat_list.insert(len(feat_list), abs(thrd_max))
            if select_feat[5] == 1:
                feat_list.insert(len(feat_list), thrd_max_ind)

        # PARAMETER 7: differential of CGM data (GET MAX VALUE)
        if select_feat[6] == 1:
            feat_list.insert(len(feat_list), np.nanmax(np.diff(row)))

        # PARAMETER 8: double differential of CGM data
        if select_feat[7] == 1:
            feat_list.insert(len(feat_list), np.nanmax(np.diff(row, n=2)))

        # add updated feature to feat_mat_list
        feat_mat_list.append(feat_list)

    return np.array(feat_mat_list)

# SSE of a data matrix (representing a cluster)
def sseFunc(data_mat_):
    # get mean along axis=0
    clust_cent = np.mean(data_mat_ ,axis=0)

    # this computes the SSE given the points and the mean of them as clust_cent
    return sum(np.linalg.norm(data_mat_-clust_cent, axis=1)**2)

# BISECTING K-MEANS UNTIL REQ_K NUMBER OF CLUSTERS
# paramter 1 has NOISELESS - feature data (FT) + ground_truth (GT) (number of columns = Fl+1)
# paramter 2 contains the labels given by DBSCAN
# paramter 3 contains required number of clusters to bisect into
def bis_KM_till_k(dataFT_GT, lbls_DB, req_k):
    
    # get the unique values of the labels
    cur_lbls = lbls_DB
    lbl_uvals = np.unique(cur_lbls)

    # get current number of clusters
    cur_num_clust = len(lbl_uvals)

    # index data according to label to get the clusters - list of 2D arrays
    clust_dataFT_GT = [ dataFT_GT[np.where(cur_lbls==lbl)] for lbl in lbl_uvals ]

    # save the SSE calculated over the loops
    # go through each label, index that data and get SSE
    # NOTE: only pass the feature matrix section to the SSE function
    sse_list = [ sseFunc(cl_d[:,:-1]) for cl_d in clust_dataFT_GT ]

    # stopping condition is until we have total req_k amount of clusters
    while cur_num_clust < req_k:
        # get the index of maximum SSE and use it to get the data to bisect("bis")
        maxSSE_ind = np.argmax(sse_list)
        bis_dataFT_GT = clust_dataFT_GT[maxSSE_ind]

        # peform bisection on the feature matrix section of the bis_data
        mod_BKM = KMeans(n_clusters=2).fit(bis_dataFT_GT[:,:-1])

        # RAISE ERROR: if the max iterations is reached
        assert mod_BKM.n_iter_ < mod_BKM.max_iter, "BIS-KMEANS ERROR: took more than max_iters (no convergence)!"

        # the labels given by B-KMeans
        labels_BKM = mod_BKM.labels_

        # from the bisection we put label 0 values at maxSSE_ind index and at same index in clust_data ...
        lbl_0_data = bis_dataFT_GT[np.where(labels_BKM==0)]
        lbl_0_sse = sseFunc(lbl_0_data[:,:-1])
        sse_list[maxSSE_ind] = lbl_0_sse
        clust_dataFT_GT[maxSSE_ind] = lbl_0_data

        # ... and add label 1 values at end of the sse_list and clust_data
        lbl_1_data = bis_dataFT_GT[np.where(labels_BKM==1)]
        lbl_1_sse = sseFunc(lbl_1_data[:,:-1])
        sse_list.append(lbl_1_sse)
        clust_dataFT_GT.append(lbl_1_data)

        # increment number of clusters
        cur_num_clust = cur_num_clust + 1

    return clust_dataFT_GT

# converts clusters to matrix comparing with the ground truths
def clustCompConv(clust_list_, req_k):
    # storing the matrix of cluster count in each GT label
    comp_mat = []
    for clust_ in clust_list_:
        # ground truth labels
        clust_GT = clust_[:,-1]
        # get number of b0, ... bN-1 frequency in clust_GT (N = required number of clusters = req_k)
        comp_count = []
        for bi in range(req_k):
            bi_count = sum(clust_GT == bi)
            comp_count.append(bi_count)

        # add this list of counts to main comp_mat
        comp_mat.append(comp_count)

    return np.array(comp_mat)

# ENTROPY function
def entropyOfClusters(clust_comp_mat_):
    # sum of elements in each row (=number of elements in the cluster represented in that row)
    num_elem_per_cluster_ = np.sum(clust_comp_mat_, axis=1, keepdims=True)

    # calculate fraction of total for each elemnt
    frac_ = clust_comp_mat_/num_elem_per_cluster_
    # NOTE: here division by zero - not possible
    # That would mean 0 elements in a cluster (not possible)

    # calculate -ve log of frac (entropy formula has a negative sign)
    # this ignores 0 when computing natural log (doesnt matter because it will get multiplies by 0 later)
    log_frac_ = -np.log(frac_, where=frac_!=0)

    # get frac times log frac
    frac_times_log_frac_ = np.multiply(frac_, log_frac_)

    # entropy for each row (each cluster)
    clust_ent_ = np.sum(frac_times_log_frac_, axis=1)

    # weight each cluster entropy with number of elements in cluster
    weight_clust_ent_ = np.multiply(num_elem_per_cluster_.ravel(), clust_ent_)

    # total entropy of entire matrix is
    tot_ent_ = sum(weight_clust_ent_)/np.sum(clust_comp_mat_)

    return tot_ent_

# PURITY function
def purityOfClusters(clust_comp_mat_):
    # get max for each row
    max_per_row_ = clust_comp_mat_.max(axis=1)

    # add the max values
    sum_maxes_ = sum(max_per_row_)

    # divide by the total number of samples
    tot_pur_ = sum_maxes_/np.sum(clust_comp_mat_)

    return tot_pur_

if __name__ == "__main__":
    main()