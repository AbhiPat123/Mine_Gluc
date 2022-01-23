# RUN COMMAND: python main.py ./Project-1-Files/CGMData.csv ./Project-1-Files/InsulinData.csv

# main.py
import sys
import numpy as np
import pandas as pd
#import sklearn
#import pickle

def main():
    # get first input data file
    dataFile1 = sys.argv[1]
    # get second input data file
    dataFile2 = sys.argv[2]

    # it is one of the files from CGMData.csv or InsulinData.csv
    if 'CGMData.csv' in dataFile1:
        cgmDataFile = dataFile1
        insDataFile = dataFile2
    else:
        cgmDataFile = dataFile2
        insDataFile = dataFile1

    # read data from the files - parse Date and Time columns as DateTime
    cgmData = pd.read_csv(cgmDataFile, parse_dates=[['Date', 'Time']])
    insData = pd.read_csv(insDataFile, parse_dates=[['Date', 'Time']])

    # focus only on datetime and Sensor Glucose data columns
    cgmData = cgmData[['Date_Time', 'Sensor Glucose (mg/dL)']]

    # create 6 columns specifying each type of hyperglycemia ranges for that row's CGM value
    # Different types of Hyperglycemia ranges
    HGC_Ranges = ['HGC', 'HGC_C', 'HGC_R1', 'HGC_R2', 'HGC_L1', 'HGC_L2']

    # TYPE 1: Hyperglycemia (HGC)
    cgmData[HGC_Ranges[0]] = cgmData['Sensor Glucose (mg/dL)'] > 180
    # TYPE 2: Hyperglycemia Critical (HGC_C)
    cgmData[HGC_Ranges[1]] = cgmData['Sensor Glucose (mg/dL)'] > 250
    # TYPE 3: Hyperglycemia Range 1 (HGC_R1)
    cgmData[HGC_Ranges[2]] = cgmData['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')
    # TYPE 4: Hyperglycemia Range 2 (HGC_R2)
    cgmData[HGC_Ranges[3]] = cgmData['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')
    # TYPE 5: Hyperglycemia Level 1 (HGC_L1)
    cgmData[HGC_Ranges[4]] = cgmData['Sensor Glucose (mg/dL)'] < 70
    # TYPE 6: Hyperglycemia Level 2 (HGC_L2)
    cgmData[HGC_Ranges[5]] = cgmData['Sensor Glucose (mg/dL)'] < 54

    # look for 'AUTO MODE ACTIVE PLGM OFF' in insData and get corresponding Date_Time
    # D_T_A for Date, Time, Alarm
    insData_D_T_A = insData[['Date_Time', 'Alarm']]
    # get the 'AUTO MODE ACTIVE PLGM OFF' date and timestamp
    autoOnData = insData_D_T_A[insData_D_T_A['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']
    # get the earliest Date_Time value index
    autoOnIndex = autoOnData['Date_Time'].idxmin()
    # index the earliest DateTime
    autoOnDateTime = insData_D_T_A.iloc[autoOnIndex]['Date_Time']

    # handle missing/NaN values
    # filters out days which have less than 80% of 288 or more than 288 amount of data
    cgmData['Valid_Day'] = cgmData.groupby(pd.Grouper(key='Date_Time', freq='1D'))['Sensor Glucose (mg/dL)'].transform(lambda x: True if (x.count() >= 231 and x.count() <= 288) else False)
    cgmData_filt = cgmData[cgmData['Valid_Day']].copy(deep=True)

    # get corresponding Manual/Auto data based on the autoOnDateTime
    # times BEFORE autoOnDateTime are MANUAL MODE
    cgmData_D_T_R_Man = cgmData_filt[ cgmData_filt['Date_Time'] < autoOnDateTime ].copy(deep=True)
    # times AFTER/ON autoOnDateTime are AUTO MODE
    cgmData_D_T_R_Aut = cgmData_filt[ cgmData_filt['Date_Time'] >= autoOnDateTime ].copy(deep=True)
    # quick check to see if data extracted is labelled correctly
    #if (cgmData_D_T_R_Man['Date_Time'].max() <= autoOnDateTime < cgmData_D_T_R_Aut['Date_Time'].min()) is False:
    #    print("ERROR: The extracted parts Manual and Auto DO NOT make sense chronologically.")

    # extract data for different segments and sub-segments of the day
    # MANUAL data
    cgm_OvrNight_Man = cgmData_D_T_R_Man[(cgmData_D_T_R_Man['Date_Time'].dt.hour >= 0) & 
                      (cgmData_D_T_R_Man['Date_Time'].dt.hour < 6)].copy(deep=True)
    cgm_DayTime_Man = cgmData_D_T_R_Man[(cgmData_D_T_R_Man['Date_Time'].dt.hour >= 6) & 
                      (cgmData_D_T_R_Man['Date_Time'].dt.hour <= 23)].copy(deep=True)
    cgm_WholeDay_Man = cgmData_D_T_R_Man.copy(deep=True)

    # AUTO data
    cgm_OvrNight_Aut = cgmData_D_T_R_Aut[(cgmData_D_T_R_Aut['Date_Time'].dt.hour >= 0) & 
                      (cgmData_D_T_R_Aut['Date_Time'].dt.hour < 6)].copy(deep=True)
    cgm_DayTime_Aut = cgmData_D_T_R_Aut[(cgmData_D_T_R_Aut['Date_Time'].dt.hour >= 6) & 
                      (cgmData_D_T_R_Aut['Date_Time'].dt.hour <= 23)].copy(deep=True)
    cgm_WholeDay_Aut = cgmData_D_T_R_Aut.copy(deep=True)

    # check if all the rows are extracted
    #if ( (len(cgm_OvrNight_Man)+len(cgm_DayTime_Man) != len(cgm_WholeDay_Man)) or
    #     (len(cgm_OvrNight_Aut)+len(cgm_DayTime_Aut) != len(cgm_WholeDay_Aut)) ):
    #    print("ERROR: The length of extracted subsegments of data DP NOT match.")

    # use groupby and count on overnight, daytime and whole day segments 
    # (Work on Each metric one-by-one through a loop)
    # two empty lists to store manual and auto values
    # for each of the two the value at index 0 is OVERNIGHT HGC (CGM > 180)
    #                                  index 1 is OVERNIGHT HGC_C (CGM > 250)
    #                                  index 2 is OVERNIGHT HGC_R1 (CGM between 70 and 180 inclusive)
    #                                  index 3 is OVERNIGHT HGC_R2 (CGM between 70 and 150 inclusive)
    #                                  index 4 is OVERNIGHT HGC_L1 (CGM < 70)
    #                                  index 5 is OVERNIGHT HGC_L2 (CGM < 54)
    # indices 6 to 11 are same ranges but for DAYTIME
    # indices 12 to 17 are same ranges but for WHOLE DAY
    man_hgc_vals = []
    aut_hgc_vals = []

    # inserting OVERNIGHT values
    for hgc_type in HGC_Ranges:
        # MANUAL
        ovrNight_Man_Perc = cgm_OvrNight_Man.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        ovrNight_Man_Avg = np.mean(ovrNight_Man_Perc)
        #print("MANUAL OVERNIGHT "+hgc_type+" = "+str(ovrNight_Man_Avg))
        # add to the list
        man_hgc_vals.append(ovrNight_Man_Avg)
        
        # AUTO
        ovrNight_Aut_Perc = cgm_OvrNight_Aut.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        ovrNight_Aut_Avg = np.mean(ovrNight_Aut_Perc)
        #print("AUTO OVERNIGHT "+hgc_type+" = "+str(ovrNight_Aut_Avg))
        # add to the list
        aut_hgc_vals.append(ovrNight_Aut_Avg)

    # inserting DAYTIME values
    for hgc_type in HGC_Ranges:
        # MANUAL
        dayTime_Man_Perc = cgm_DayTime_Man.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        dayTime_Man_Avg = np.mean(dayTime_Man_Perc)
        #print("MANUAL DAYTIME "+hgc_type+" = "+str(dayTime_Man_Avg))
        # add to the list
        man_hgc_vals.append(dayTime_Man_Avg)
        
        # AUTO
        dayTime_Aut_Perc = cgm_DayTime_Aut.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        dayTime_Aut_Avg = np.mean(dayTime_Aut_Perc)
        #print("AUTO DAYTIME "+hgc_type+" = "+str(dayTime_Aut_Avg))
        # add to the list
        aut_hgc_vals.append(dayTime_Aut_Avg)

    # inserting WHOLE DAY values
    for hgc_type in HGC_Ranges:
        # MANUAL
        wholeDay_Man_Perc = cgm_WholeDay_Man.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        wholeDay_Man_Avg = np.mean(wholeDay_Man_Perc)
        #print("MANUAL WHOLE DAY "+hgc_type+" = "+str(wholeDay_Man_Avg))
        # add to the list
        man_hgc_vals.append(wholeDay_Man_Avg)
        
        # AUTO
        wholeDay_Aut_Perc = cgm_WholeDay_Aut.groupby(pd.Grouper(key='Date_Time', freq='1D'))[hgc_type].apply(lambda x: ((sum(x)/288)*100) )
        # average over all valid days
        wholeDay_Aut_Avg = np.mean(wholeDay_Aut_Perc)
        #print("AUTO WHOLE DAY "+hgc_type+" = "+str(wholeDay_Aut_Avg))
        # add to the list
        aut_hgc_vals.append(wholeDay_Aut_Avg)

    # create a 2x18 matrix from the two lists
    results_matrix = np.row_stack((man_hgc_vals, aut_hgc_vals))

    # create a CSV file of the results
    np.savetxt('Results.csv', results_matrix, delimiter=',')

    #print("Results.csv file created!")
    pass

if __name__ == "__main__":
    main()