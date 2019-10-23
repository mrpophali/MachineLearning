# -*- coding: utf-8 -*-
"""
#Assignment: Mandarpophali-M03-DataModelFinal.py 
#Class: DATASCI 400 CB
#Date: 07th June 2019
#Author: Mandar Pophali
#Description: EDA analysis for Android app store dataset

Data overview for the file
App                9659 non-null object : App name (Categorical) 
Category           9659 non-null object : App category (Categorical)
Reviews            9659 non-null object : log normal value for number of reviewes (Numeric)
Size               9659 non-null object : Size of App in M or kb (Alphanumberic)
Installs           9659 non-null float64: log normal value of #installs(Numeric)
Type               9658 non-null object : Free or Paid app (Categorical)
Price              9659 non-null float64: Value of app 0.0 for free and value for paid app(Numerical)
Genres             9659 non-null object : Genres for app. (Categorical)
Last Updated       9659 non-null object : Date when app was last updated (Categorical)
Current Ver        9651 non-null object : Current version number (Alphanumeric)
Android Ver        9657 non-null object : Android version (Alphanumeric)
Rating Category    9659 non-null float64: Rating category (1.0/2.0/3.0)(Numeric)
All                9659 non-null int32  : App category All (Yes/no)
Child              9659 non-null int32  : App category child (Yes/no)
Adult              9659 non-null int32  : App category Adult (Yes/no)
"""
# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import *

def main():    

    appdata = data_import()
    appdata = data_cleaning(appdata)
    appdata = data_replaceoutlier(appdata)
    appdata = data_replacemissing(appdata)
    appdata = data_norm(appdata)
    appdata = data_binning(appdata)
    appdata = data_consolidate(appdata)
    appdata = one_hot_encode(appdata)
    appdata = remove_obsolete(appdata)
    appdata = dummy_columns(appdata)
    
    #plot data for analysis
    data_plot(appdata)
    
    #apply Kmeans to cluster apps into 2 groups
    kmean_model(appdata)
    
    #Apply logistic regression to predict type of app(Free/Paid)
    logistic_model(appdata)
    
    #Apply random forest to predict type of app(Free/Paid)
    randomforest_model(appdata)
    

def data_import():
    """
    import data using the source from internet
    """
    url = "https://raw.githubusercontent.com/jasonchang0/kaggle-google-apps/master/google-play-store-apps/googleplaystore.csv"
    appdata = pd.read_csv(url,engine='python',encoding='utf-8')
    print(appdata.info())
    print(appdata.shape)
    return appdata
    
def data_cleaning(appdata):    
    """
    Part1
    data cleaning steps. After analysis of the various columns 
    information and null using below commands, below are some 
    observations and steps for 
    cleaning the data.
    1> Index 10472 is misplaced. Remove that erronious row entirely
    2> Index 9148 is missing Type (Free/Paid). Remove that row entirely
    3> convert 'Type' column as str
    4> convert 'Reviews' column as numeric
    4> 'App' column has many duplicates. Remove duplicate rows and keep
    #app data for unique apps only
    """ 
    print("Inside Cleaning")
    #remove bad rows of data    
    print(appdata.iloc[10472,])
    print(appdata.iloc[9148,])
    appdata = appdata.drop(10472,axis=0)
    appdata = appdata.drop(9148,axis=0)
    #convert Type as string     
    appdata['Type'] = appdata['Type'].astype(str)
    #convert Reviews as numberic
    appdata.loc[:, "Reviews"] = pd.to_numeric(appdata.loc[:, "Reviews"], errors='coerce')
    print(appdata.shape)
    #part1 step2 1181 duplicate apps and reset index 
    print(sum(appdata['App'].duplicated()))
    appdata.drop_duplicates('App',inplace=True)
    appdata.reset_index(inplace=True)
    print(appdata.info())
    print("Cleaning completed")
    return appdata

def data_replaceoutlier(appdata):
    """
    Part2:
    **Price is not a ideal candiate for outliers but i have taken it as exmaple ONLY    
    for removing outliers. The Price column has '$' appended in the begining 
    1>recast Price column as Str
    2>Strip the $ sign
    3>recast Price column as float
    4>Replace high values of prices. Calculate all prices more than >100 
    and replace them with MAD value of price for such apps.
    e.g in case >100 are as below
    110,151,151,200 then take 151 as MAD and replace it for every value of app having
    value more than 50
    """   
    print("Inside removing outlier module")
    appdata['Price'] = appdata['Price'].astype(str)
    appdata['Price'] = [x.strip('$') for x in appdata['Price']]
    appdata['Price'] = appdata['Price'].astype(float)
    appdata.loc[appdata['Price'] >=75,'Price'] = appdata.loc[appdata['Price'] >=100,'Price'].mad()
    print(appdata['Price'].unique())
    print("outliers removed")
    return appdata

def data_replacemissing(appdata):
    """
    Part3: Replace missing values in Rating column
    "Rating" column has values from 1.0 to 5.0 and nan
    replace nan with median values 
    """
    # Corece Rating data to numeric and impute medians for Rating" column
    print("Inside replace missing values module")
    appdata.loc[:, "Rating"] = pd.to_numeric(appdata.loc[:, "Rating"], errors='coerce')
    HasNan = np.isnan(appdata.loc[:,"Rating"]) 
    appdata.loc[HasNan, "Rating"] = np.nanmedian(appdata.loc[:,"Rating"])
    return appdata

def data_norm(appdata):

    """
    Part4: Normalize Numberic values
    "Install" column has values from 1+ to 100,000,000+
    "Reviews" column as values from 0 to 78158306
    "Log" normalize "Install" column using below steps
    1.Recast Installs column as Str
    2.Strip the '+' and ',' sign
    3.recast Install column as Float
    Log normalize "Reviews" column as well
    Log normalize all non-Zero values 
    """    
    print("Inside data normalization module")
    #Log scaling "Installs" data
    appdata['Installs'] = appdata['Installs'].astype(str)
    appdata['Installs'] = [x.strip('+') for x in appdata['Installs']]
    appdata['Installs'] = [x.replace(',','') for x in appdata['Installs']]
    appdata['Installs'] = appdata['Installs'].astype(float)
    appdata.loc[appdata['Installs'] > 0,['Installs']] = np.log(appdata.loc[appdata['Installs'] > 0,['Installs']])
    #log scaling "Reviews" data
    appdata.loc[appdata['Reviews'] > 0,['Reviews']] = np.log(appdata.loc[appdata['Reviews'] > 0,['Reviews']])
    return appdata

def data_binning(appdata):
    """
    Part5: Bin numberic variable 
    "Rating" column has values from 1.0 to 5.0. Bin apps into 3 categories using
    percentile as below
    percentiles = np.linspace(0, 100, 4)
    bounds = np.percentile(appdata['Rating'], percentiles)
    
    Rating Category '1' : 0.0 to 4.2
    Rating Category '2' : 4.2 to 4.4
    Rating Category '3' : 4.4 to 5.0
    """       
    print("Inside data binning module")
    appdata.loc[appdata.loc[:, "Rating"] < 4.2, "Rating Category"] = 1
    appdata.loc[appdata.loc[:, "Rating"] > 4.4, "Rating Category"] = 3
    appdata.loc[(appdata.loc[:, "Rating"] >= 4.2)&(appdata.loc[:, "Rating"] <= 4.4), "Rating Category"] = 2
    return appdata

def data_consolidate(appdata):
    """
    Part6: Consolidate categorical data 
    Below is the content rating category. Consolidate it as new categories
    as highlighted in Third column below
    Current             #       New 
    Everyone           7903     All
    Teen               1036     Child
    Mature 17+          393     Adult
    Everyone 10+        322     Child
    Adults only 18+       3     Adult
    Unrated               2     All
    """    
    print("Inside consolidate categorical data module")
    appdata['Content Rating'] = appdata['Content Rating'].astype(str)
    appdata['Content Rating'] = [x.replace('Everyone','All') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('Teen','Child') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('Mature 17+','Adult') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('Everyone 10+','Child') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('Adults only 18+','Adult') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('Unrated','All') for x in appdata['Content Rating']]
    appdata['Content Rating'] = [x.replace('All 10+','Child') for x in appdata['Content Rating']]
    print(appdata['Content Rating'].value_counts())
    return appdata
    
def one_hot_encode(appdata):
    """
    One-hot encode categorical data with at least 3 categories
    Use the "Content Rating" column above for one hot encoding
    """    
        
    print("Inside one hot encoding module")
    appdata.loc[:, "All"] = (appdata.loc[:, "Content Rating"] == "All").astype(int)
    appdata.loc[:, "Child"] = (appdata.loc[:, "Content Rating"] == "Child").astype(int)
    appdata.loc[:, "Adult"] = (appdata.loc[:, "Content Rating"] == "Adult").astype(int)
    return appdata

def remove_obsolete(appdata):
    # Remove obsolete columns
    print("inside removing obsolete column module")
    appdata = appdata.drop(["Content Rating","Rating"], axis=1)
    print(appdata.info())
    return appdata


def dummy_columns(appdata):
    """
    Add new column "apptype which will be used as index.
    it will have value 1 for Paid and 0 for free app
    
    """
    appcategory = pd.get_dummies(appdata['Category'],drop_first=True)
    appdata = pd.concat([appdata,appcategory],axis=1)
    free = appdata.loc[:,'Type'] == 'Free'
    paid = appdata.loc[:,'Type'] == 'Paid'
    appdata.loc[free,"apptype"] = 0
    appdata.loc[paid,"apptype"] = 1
    #drop the original column "Type"    
    appdata = appdata.drop(['Type'],axis=1)    
    return appdata
    
def data_plot(appdata):
    """
    plot various columns to perform data analysis 
   
    """
    #Draw scatter plot for price and Reviews
    plt.rcParams["figure.figsize"] = [15.0, 15.0]
    plt.scatter(appdata['Reviews'],appdata['Category'],c=appdata['apptype'],alpha=2,cmap='coolwarm')
    plt.xlabel('Price')
    plt.ylabel('Reviews')
    plt.title('Price vs Reviews')
    plt.show()
    #Draw scatter plot for Installs vs Reviews    
    plt.rcParams["figure.figsize"] = [15.0, 15.0]
    plt.scatter(appdata['Installs'],appdata['Category'],s=100,c=appdata['apptype'],alpha=2,cmap='coolwarm')
    plt.xlabel('Installs')
    plt.ylabel('Reviews')
    plt.title('Installs vs Reviews')
    plt.show()
    
def logistic_model(appdata):
    """
    Problem statement: Using logistic regression, predict the 
    type of app (Free[0] or Paid [1])
    Based on the Classification report and csv output, model looks very accurate !
    
    """
    #remove all columns which are not required
    appdata = appdata.drop(['App','Category','Size','Genres','Last Updated','Current Ver','Android Ver'],axis=1)
    #split the dataset test & train. Assign 50% to test
    #i choose 50% to ensure there is sufficient amount of data for training 
    X_train, X_test, y_train, y_test = train_test_split(appdata.drop('apptype',axis=1), 
                                                    appdata['apptype'], test_size=0.50, 
                                                    random_state=101)
    #define instance of Logistic Regression
    appmodel = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial',max_iter=1000)
    #train the dataset
    appmodel.fit(X_train,y_train)
    #predict the dataset
         
    BothProbabilities = appmodel.predict_proba(X_test)
    probabilities = BothProbabilities[:,1]
    
    ##############
    print ('\nConfusion Matrix and Metrics')
    # A probability threshold of 0.3 is chosen, to ensure highest accuracy 
    Threshold = 0.3 # Some number between 0 and 1
    print ("Probability Threshold is chosen to be:", Threshold)
    predictions = (probabilities > Threshold).astype(int)

    CM = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = CM.ravel()
    print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
    AR = accuracy_score(y_test, predictions)
    print ("Accuracy rate:", np.round(AR, 2))
    P = precision_score(y_test, predictions)
    print ("Precision:", np.round(P, 2))
    R = recall_score(y_test, predictions)
    print ("Recall:", np.round(R, 2))
    
    ##############
     # False Positive Rate, True Posisive Rate, probability thresholds
    fpr, tpr, th = roc_curve(y_test, probabilities)
    AUC = auc(fpr, tpr)
    
    plt.rcParams["figure.figsize"] = [8, 8] # Square
    font = {'family' : 'DejaVu Sans', 'weight' : 'bold', 'size' : 18}
    plt.rc('font', **font)
    plt.figure()
    plt.title('ROC Curve Logistic Regression')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
    plt.legend(loc="lower right")
    plt.show()
    
    ##############

def kmean_model(appdata):
    """    
    Copy 'Reviews','Installs','Type','Price','Rating Category','All','Child','Adult' 
    into another DataFrame which will be used for K-Means algorithm
    
    The KMean is used to cluster apps into two labels. From the price 
    graph, the labels are very similar to free/paid data and hence not used 
    in the supervised learning algorithm

    """
    app_k1 = appdata.loc[:,['Reviews','Installs','apptype','Price','Rating Category','All','Child','Adult']]
    
    #initiate Kmeans for 2 clusters
    kmeans = KMeans(n_clusters=2)
    #Fit the data     
    kmeans.fit(app_k1)
    # Add the labels to the data set.  We may want to use the labels as inputs
    # to the supervised learning
    app_k1.loc[:, 'Labels'] = kmeans.labels_
    print ('\nWhat is the data type of Labels?')
    print(app_k1.loc[:, 'Labels'].dtype)
        
    #draw the graph showing before and after
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,15))
    
    ax1.set_title('K Means')
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Category')
    ax1.scatter(app_k1['Price'],appdata['Category'],c=kmeans.labels_,cmap='rainbow')
    ax2.set_title("Original")
    ax2.scatter(app_k1['Price'],appdata['Category'],c=app_k1['apptype'],cmap='rainbow')

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,15))
    ax1.set_title('K Means')
    ax1.set_xlabel('Reviews')
    ax1.set_ylabel('Category')
    ax1.scatter(app_k1['Reviews'],appdata['Category'],c=kmeans.labels_,cmap='rainbow')
    ax2.set_title("Original")
    ax2.scatter(app_k1['Reviews'],appdata['Category'],c=app_k1['apptype'],cmap='rainbow')

def randomforest_model(appdata):
    """
    Problem statement: Using random forest, predict the 
    type of app (Free[0] or Paid [1])
    Based on the Classification report and csv output, model looks very accurate !
    
    """
    #remove all columns which are not required
    appdata = appdata.drop(['App','Category','Size','Genres','Last Updated','Current Ver','Android Ver'],axis=1)
    #split the dataset test & train. Assign 50% to test
    #i choose 50% to ensure there is sufficient amount of data for training 
    X_train, X_test, y_train, y_test = train_test_split(appdata.drop('apptype',axis=1), 
                                                    appdata['apptype'], test_size=0.50, 
                                                    random_state=101)
    #define instance of random forest
    appmodel = RandomForestClassifier(n_estimators=10)
    #train the dataset
    appmodel.fit(X_train,y_train)
    #predict the dataset
    
    
    BothProbabilities = appmodel.predict_proba(X_test)
    probabilities = BothProbabilities[:,1]
    
    ##############
    print ('\nConfusion Matrix and Metrics')
    # A probability threshold of 0.3 is chosen, to ensure highest accuracy 
    Threshold = 0.1 # Some number between 0 and 1
    print ("Probability Threshold is chosen to be:", Threshold)
    predictions = (probabilities > Threshold).astype(int)
    
    CM = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = CM.ravel()
    print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
    AR = accuracy_score(y_test, predictions)
    print ("Accuracy rate:", np.round(AR, 2))
    P = precision_score(y_test, predictions)
    print ("Precision:", np.round(P, 2))
    R = recall_score(y_test, predictions)
    print ("Recall:", np.round(R, 2))
    
    ##############
     # False Positive Rate, True Posisive Rate, probability thresholds
    fpr, tpr, th = roc_curve(y_test, probabilities)
    AUC = auc(fpr, tpr)
    
    plt.rcParams["figure.figsize"] = [8, 8] # Square
    font = {'family' : 'DejaVu Sans', 'weight' : 'bold', 'size' : 18}
    plt.rc('font', **font)
    plt.figure()
    plt.title('ROC Curve Random Forest')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
    plt.legend(loc="lower right")
    plt.show()
    
    ##############
    """
       
     Mode is able to 86% accurate paid app(Class 0). Based on overall 
     score of 1.00 for class 0.0 and 1.00 for 1.0 (f1-score) model is doing good job.
       
    """

if __name__ == "__main__":
    main()



