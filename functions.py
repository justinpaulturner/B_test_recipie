import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from textwrap import wrap
from scipy.stats import mannwhitneyu as MannWhitneyU
import os
from os import path


def set_dataframes():
    #rename xlsx file
    os.rename('Evolytics Data Science Exercise_with key final 10 18 17.xlsx','data.xlsx') 


    # # Set pandas to display all columns and rows.
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)


    # Importing data from xlsx file sheets 1 & 2.
    df = pd.read_excel('data.xlsx', sheet_name=0)
    key_df = pd.read_excel('data.xlsx', sheet_name=1)


    # Replacing '(null)' string with NaN's.
    df = df.replace('(null)', np.nan)


    # Changing columns 21,22,23,24 to floats.
    df.iloc[:, 21:25] = df.iloc[:, 21:25].astype('float64')


    # Adding a total revenue column to df.
    df['total_revenue'] = df.iloc[:, 21:25].sum(axis=1)

    # Adding a standard purchase column to df.
    df['standard_purchase'] = df['purchase_flag']*(1-df['upgrade_and_purchase']) # 1 where purchased is 1 and premium is 0

    # Reverting products 1 - 4 back to null strigns
    df.iloc[:, 13:17] = df.iloc[:, 13:17].fillna('null_str')

    # Adding columns for purchased product type
    products = ['HIKE','COOK','FOOD','SOCKS','WTR','BAG','LEADER','RUN','MULTISLP','PERF']

    for prod in products:
        df[f'bought_{prod}'] = df.iloc[:, 13:17].applymap(lambda x: prod in x).any(1).astype(int)

    # Change revenue nans to 0 integers and units nans to 0 intigers
    df.iloc[:, 17:25] = df.iloc[:, 17:25].fillna(0)

    # Add dummy columns for os family and state to df
    os_dummies = pd.get_dummies(df['operating_system_family'])
    df = pd.concat([df, os_dummies], axis=1, sort=False)
    state_dummies = pd.get_dummies(df['user_State'])
    df = pd.concat([df, state_dummies], axis=1, sort=False)


    # Saving DataFrames as csv files.
    df.to_csv('data.csv')
    key_df.to_csv('key.csv')


    # Reading CSV's as DataFrames (load faster than xlsx)
    df = pd.read_csv('data.csv', index_col = 0)
    key_df = pd.read_csv('key.csv', index_col = 0)
    
    return df, key_df

def total_premium_conversion_rate_binary(column):
    numerator_belongs = df[(df[column]==1)&(df['upgrade_and_purchase']==1)].shape[0]
    denominator_belongs = df[(df[column]==1)].shape[0]
    numerator_not_belongs = df[(df[column]==0)&(df['upgrade_and_purchase']==1)].shape[0]
    denominator_not_belongs = df[(df[column]==0)].shape[0]
    if denominator_belongs == 0: belongs = 0 
    else: belongs = (numerator_belongs/denominator_belongs)
    if denominator_not_belongs == 0: not_belongs = 0 
    else: not_belongs = (numerator_not_belongs/denominator_not_belongs)
    
    return belongs, not_belongs

def total_premium_conversion_rate_binary_pval(column):
    belongs = df[(df[column]==1)]['upgrade_and_purchase']
    not_belongs = df[(df[column]==0)]['upgrade_and_purchase']
    pval = MannWhitneyU(belongs,not_belongs)[1]
    return pval

def said_yes_conversion_rate_binary(column):
    numerator_belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==1)&(df['upgrade_and_purchase']==1)].shape[0]
    denominator_belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==1)].shape[0]
    numerator_not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==1)&(df['upgrade_and_purchase']==1)].shape[0]
    denominator_not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==1)].shape[0]
    if denominator_belongs == 0: belongs = 0 
    else: belongs = (numerator_belongs/denominator_belongs)
    if denominator_not_belongs == 0: not_belongs = 0 
    else: not_belongs = (numerator_not_belongs/denominator_not_belongs)
    
    return belongs, not_belongs

def said_yes_conversion_rate_binary_pval(column):
    belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==1)]['upgrade_and_purchase']
    not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==1)]['upgrade_and_purchase']
    pval = MannWhitneyU(belongs,not_belongs)[1]
    return pval

def total_conversion_rate_binary(column):
    numerator_belongs = df[(df[column]==1)&(df['purchase_flag']==1)].shape[0]
    denominator_belongs = df[(df[column]==1)].shape[0]
    numerator_not_belongs = df[(df[column]==0)&df['purchase_flag']==1].shape[0]
    denominator_not_belongs = df[(df[column]==0)].shape[0]
    if denominator_belongs == 0: belongs = 0 
    else: belongs = (numerator_belongs/denominator_belongs)
    if denominator_not_belongs == 0: not_belongs = 0 
    else: not_belongs = (numerator_not_belongs/denominator_not_belongs)
        
    return belongs, not_belongs 

def total_conversion_rate_binary_pval(column):
    belongs = df[(df[column]==1)]['purchase_flag']
    not_belongs = df[(df[column]==0)]['purchase_flag']
    pval = MannWhitneyU(belongs,not_belongs)[1]
    return pval

def said_no_conversion_rate_binary(column):
    numerator_belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==0)&(df['standard_purchase']==1)].shape[0]
    denominator_belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==0)].shape[0]
    numerator_not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==0)&(df['standard_purchase']==1)].shape[0]
    denominator_not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==0)].shape[0]
    if denominator_belongs == 0: belongs = 0 
    else: belongs = (numerator_belongs/denominator_belongs)
    if denominator_not_belongs == 0: not_belongs = 0 
    else: not_belongs = (numerator_not_belongs/denominator_not_belongs)
    
    return belongs, not_belongs

def said_no_conversion_rate_binary_pval(column):
    belongs = df[(df[column]==1)&(df['yes_upgrade_flag']==0)]['standard_purchase']
    not_belongs = df[(df[column]==0)&(df['yes_upgrade_flag']==0)]['standard_purchase']
    pval = MannWhitneyU(belongs,not_belongs)[1]
    return pval

def binary_conversion_comparisons(column):
    return [total_premium_conversion_rate_binary(column),
          said_yes_conversion_rate_binary(column),
          total_conversion_rate_binary(column),
          said_no_conversion_rate_binary(column),
          total_premium_conversion_rate_binary_pval(column),
          said_yes_conversion_rate_binary_pval(column),
          total_conversion_rate_binary_pval(column),
          said_no_conversion_rate_binary_pval(column)]

def plot_comparisons(column):
    
    
    names = [f'{column}',f'Non {column}']
    values = binary_conversion_comparisons(column)[:4]
    p_vals = binary_conversion_comparisons(column)[4:]
    titles = [f"Premier CR of {column} vs Non {column}",
              f"Premier CR of {column} vs Non {column} Who Selected 'YES'",
              f"Total CR: Users That Made a Purchase",
              f"Standard CR of {column} vs Non {column} Who Selected 'NO'"]

    fig, axs = plt.subplots(2,2, figsize=(20, 8), sharey=True)

    axs[0,0].bar(names, values[0])
    axs[0,1].bar(names, values[1])
    axs[1,0].bar(names, values[2])
    axs[1,1].bar(names, values[3])
    
    if p_vals[0] <0.001:
        axs[0,0].annotate(f"p value: \n<0.001",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    else: axs[0,0].annotate(f"p value: \n{round(p_vals[0], 5)}",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    if p_vals[1] <0.001:
        axs[0,1].annotate(f"p value: \n<0.001",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    else: axs[0,1].annotate(f"p value: \n{round(p_vals[1], 5)}",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    if p_vals[2] <0.001:
        axs[1,0].annotate(f"p value: \n<0.001",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    else: axs[1,0].annotate(f"p value: \n{round(p_vals[2], 5)}",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    if p_vals[3] <0.001:
        axs[1,1].annotate(f"p value: \n<0.001",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    else: axs[1,1].annotate(f"p value: \n{round(p_vals[3], 5)}",xy=(0.425,-.2),xycoords='axes fraction',fontsize=14)
    
    axs[0,0].set_title("\n".join(wrap(titles[0], 35)), fontsize = 30)
    axs[0,1].set_title("\n".join(wrap(titles[1], 35)), fontsize = 30)
    axs[1,0].set_title("\n".join(wrap(titles[2], 35)), fontsize = 30)
    axs[1,1].set_title("\n".join(wrap(titles[3], 35)), fontsize = 30)

    axs[0,0].tick_params(axis="x", labelsize=22)
    axs[0,1].tick_params(axis="x", labelsize=22)
    axs[1,0].tick_params(axis="x", labelsize=22)
    axs[1,1].tick_params(axis="x", labelsize=22)
    
    axs[0,0].tick_params(axis="y", labelsize=17)
    axs[1,0].tick_params(axis="y", labelsize=17)

    plt.subplots_adjust(hspace=.95)
    return plt.show()
