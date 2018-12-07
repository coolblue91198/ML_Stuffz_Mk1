# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


# Use Seaborn data 
data = sns.load_dataset("tips")
data = data[["total_bill", "tip"]]

# Look at data layout
print(data.head())

# Statistical Analysis 
print(data.describe())

# Visual Check of Data
plt.figure()
sns.distplot(data["total_bill"])    #total_bill Histogram

plt.figure()
sns.distplot(data["tip"])   #tip Histogram

plt.figure()
sns.relplot(x="total_bill", y="tip", data=data)     #Scatter Plot 


# Find Outliers Visually
plt.figure()
sns.boxplot(data["total_bill"])     #Box-Whisker Plot (total_bill)

plt.figure()
sns.boxplot(data["tip"])    #Box-Whisker Plot (tip)


# Define Outliers 
dinfo = data.describe()

iqr_tb = dinfo["total_bill"][6] - dinfo["total_bill"][4]    #Inter-Quartile Range (total_bill)

iqr_tip = dinfo["tip"][6] - dinfo["tip"][4]    #Inter-Quartile Range (tip)

ul_tb = dinfo["total_bill"][6] + (1.5 * iqr_tb)     #Upper Limit (total_bill)

ul_tip = dinfo["tip"][6] + (1.5 * iqr_tip)     #Upper Limit (tip)

ll_tb = dinfo["total_bill"][4] - (1.5 * iqr_tb)     #Lower Limit (total_bill)

ll_tip = dinfo["tip"][4] - (1.5 * iqr_tip)     #Lower Limit (tip)


# Find Outliers
uo_tb = data["total_bill"] >= ul_tb     #Outliers above upper limit (total_bill)

uo_tip = data["tip"] >= ul_tip     #Outliers above upper limit (tip)

lo_tb = data["total_bill"] <= ll_tb     #Outliers below lower limit (total_bill)

lo_tip = data["tip"] <= ll_tip     #Outliers below lower limit (tip)

outliers_tb = uo_tb | lo_tb     #Combine upper and lower outliers

outliers_tip = uo_tip | lo_tip  #Combine upper and lower outliers 


# Remove Outliers
outlier_index_tb = list(data[outliers_tb].index.values)  #Get index of outliers (total_bill)

outlier_index_tip = list(data[outliers_tip].index.values)   #Get index of outliers (tip)

outlier_index = list(set().union(outlier_index_tb, outlier_index_tip))   #Combine outlier indices

data = data.drop(outlier_index).reset_index(drop=True)  #Drop outliers and reset the index


# Plot w/out outliers
plt.figure()
sns.distplot(data["total_bill"])    #total_bill Histogram

plt.figure()
sns.distplot(data["tip"])   #tip Histogram

plt.figure()
sns.relplot(x="total_bill", y="tip", data=data)     #Scatter Plot 

plt.figure()
sns.distplot(data["total_bill"])    #total_bill Histogram

plt.figure()
sns.distplot(data["tip"])   #tip Histogram

plt.figure()
sns.relplot(x="total_bill", y="tip", data=data)     #Scatter Plot 

plt.figure()
sns.boxplot(data["total_bill"])     #Box-Whisker Plot (total_bill)

plt.figure()
sns.boxplot(data["tip"])    #Box-Whisker Plot (tip)

