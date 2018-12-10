# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.scorer import make_scorer
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.scorer import make_scorer 
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

"""
Created on Sat Nov 24 13:28:11 2018

@author: Muhammad Shahraiz Khan Niazi and Muhammad Daniyal Saqib
"""
'''
#Start:Kaggle.com
#Explanation: Plot a bar graph against SalePrice
#End '''
def barPlot(df, var):
    
    plt.xticks(rotation =90)
    sns.barplot(df.loc[:,'SalePrice'], df.loc[:, var])
    plt.title('SalePrice vs ' + var)
    

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def colDrop(df, colName):
    return df.drop(colName, axis = 1)
    

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
'''
#Start:Kaggle.com
#Explanation: to figure out the division/scattering of the data's values
#End '''

def checkRange(df, var):
    labels = df.loc[:,var].unique()
    sizes = df.loc[:,var].value_counts().values
    percent = 100.*sizes/sizes.sum()
    labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, percent)]
    print(labels)
    
    
   

    
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
'''
#Start:Kaggle.com
#Explanation: Plot a historgram graph
#End '''

def histogram(df, var):
    sns.distplot(df.loc[:, var]);
    fig = plt.figure()

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------



def join(df1, df2):
    return pd.concat([df1, df2], axis = 1)

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
'''
#Start:Kaggle.com
#Explanation: Plot a scatter graph against SalePrice
#End '''
    
def scatterPlot(df, var):
    data = pd.concat([df.loc[:, 'SalePrice'], df.loc[:,var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#This method implements binary encoding on particular columns in the dataframe and returns that dataframe
def encode(ser):
    return pd.get_dummies(ser)


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#This function checks percentage of the number of NA values in a coloumn  
def checkMissingValues(df):
    return df.apply(lambda col: (col.isnull().sum()/ df.shape[0]) *100 )


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------


#This function reads the data from the file and returns the entire dataset, a list of input cols and outputCol 
def readData_1(numRows = None):
    
    inputCols = ['MSSubClass', 'MSZoning', 'LotFrontage',
                 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LotConfig',
                 'Neighborhood', 'Condition1', 'Condition2', 'BldgType'
                 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt'
                 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
                 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
                 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
                 'CentralAir', 'Electrical','1stFlrSF',  '2ndFlrSF', 'LowQualFinSF',
                 'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Bedroom',
                 'Kitchen','KitchenQual','TotRmsAbvGrd','Functional',
                 'Fireplaces','FireplaceQu','GarageType','GarageYrBlt',
                 'GarageFinish','GarageCars','GarageArea',
                 'GarageQual','GarageCond','PavedDrive',
                 'WoodDeckSF','OpenPorchSF','EnclosedPorch',
                '3SsnPorch', 'ScreenPorch','PoolArea',
                'PoolQC','Fence','MiscFeature', 'MiscVal',
                'MoSold','YrSold','SaleType','SaleCondition']
    
    outputCol = ['SalePrice']
    
    
    trainHouseDF = pd.read_csv('Data/train.csv')
    testHouseDF = pd.read_csv('Data/test.csv')
    
    houseDF = pd.concat([trainHouseDF, testHouseDF], sort=True)
    #houseDF = houseDF.sample(frac = 1, random_state = 99).reset_index(drop = True)
    
    return houseDF, inputCols, outputCol


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------


#This function reads the data from the file and returns the two datasets - 
# training and testing,  a list of input cols and outputCol 

def readData(numRows = None):
    trainHouseDF = pd.read_csv('Data/train.csv')
    testHouseDF = pd.read_csv('Data/test.csv')
    outputCol = ['SalePrice']

    return trainHouseDF, testHouseDF, outputCol


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------


def c2(df, colName):
    return df.loc[:, colName].corr(df.loc[:, 'SalePrice'])



#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#This function finds out the correlation between the numerical columns and the SalePrice
def corrRelationForNumerical():
    
    df, b, c = readData_1()
    corr=df.corr()["SalePrice"]
    print(corr[np.argsort(corr, axis=0)[::-1]])
    
    inputCols = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 
                 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea',
                 'Fireplaces']
    
    corrMatrix= df[inputCols].corr()
    sns.set(font_scale=1.10)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01, square=True,annot=True,cmap='viridis',linecolor="white")
    plt.title('Correlation between features')
    
    '''
    According to the correlation that we found with respect to SalePrice, we think:
    values that are above close to 0.5 or greater than 0.5 would affect SalePrice and values that are less than -0.1.
    Hence, our predictors list till now is: OverallQual, GrLiveArea, GarageCars, GarageArea
    TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, GarageYrBl
    '''

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#This function would preprocess the nonNumerical data and we would evaluate which column to keep as a parameter
#for the final algorithm 
def dataProcessing():
    
    
    df, b, c = readData()
    
    #____________________________________________________________________________________________
    #While going through the Utilities - there is only one entry that is other than AllPub (Id = 945)
    #Hence dropping ID 945 and then the entire col would not affect the TargetPrice - Sale price
    df = df.drop(labels = 944, axis = 0)
    
    '''
    
    From: Kaggle.com
    Explanation: After trying different plotting techniques, such as boxPlot, scatterplot and few others, 
    We think this describes the neighborhood perfectly. 
    '''
    
    #barPlot(df, 'Neighborhood')
    df = colDrop(df, 'Neighborhood')
    
    
    #End: Kaggle.com
    '''Other than few extra spikes, the SalePrice is not really affected that much by Neighborhood, as it is all between
    100000 - 200000. Shahraiz and I believe, the Neighborhood really doesn't matter. Hence, we would drop this column'''
    
    
    
    #----------------------------------------------------------------------------------------------------------
    #MSZoning Attribute
    #plt.xticks(rotation =90)
    #plt.scatter(df.loc[:,'MSZoning'], df.loc[:, 'SalePrice'])
    #plt.title('SalePrice vs MSZoning')
    
    '''This graph clearly shows that they majority of the data is in RL; however, to confirm it further we would use Classifcation graph
    to plot it'''
    
    #labels = df.loc[:,"MSZoning"].unique()
    #sizes = df.loc[:,"MSZoning"].value_counts().values
    #explode=[0.1,0,0,0,0]
    #parcent = 100.*sizes/sizes.sum()
    #labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]
    #colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
    #patches = plt.pie(sizes, colors=colors,explode=explode, shadow=True,startangle=90)
    
    #plt.title("Zoning Classification")
    #plt.show()
    #print(labels)
    #print()
    
    '''Clearly a large part of the pie is yellow green which is RL - and the second most occuring 
    is gold which is RM. Therefore, we would keep the values that are RL and RM and remove the rows 
    that are any other value.'''
    '''We would discreditize the data in MSZoning: 1 - RL, 0 - RM and then find the correlation'''
    
    df = df.loc[(df.loc[:,'MSZoning'] == 'RL') | (df.loc[:,'MSZoning']=='RM') | (df.loc[:,'MSZoning']=='C')]
    df.loc[:,'MSZoning'] = df.loc[:,'MSZoning'].map(lambda val: 0 if(val=='RL') else val)
    df.loc[:,'MSZoning'] = df.loc[:,'MSZoning'].map(lambda val: 1 if(val=='RM') else val)
    df.loc[:,'MSZoning'] = df.loc[:,'MSZoning'].map(lambda val: 2 if(val=='C') else val)
    #print(corrRelation2Cols(df, ['MSZoning', 'SalePrice']))
    '''
    The correlation between SalePrice and MSZoning is 0.29556497792 - that is below the threshold of 0.5; hence, we
    would not use this as well. 
    '''
    df = colDrop(df, 'MSZoning')
    
    #----------------------------------------------------------------------------------------------------------
    #Street
    
    #plt.xticks(rotation =90)
    #plt.scatter(df.loc[:,'Street'], df.loc[:, 'SalePrice'])
    #plt.title('SalePrice vs Street')
    
    df = df.loc[(df.loc[:, 'Street'] == 'Pave')]
    

    '''
    The graph shows that majority of the values are Pave.
    Therefore we would keep all the values that are Pave and get rid of the column.
    '''
    
    df = colDrop(df, 'Street')
    
    
    
    
    #----------------------------------------------------------------------------------------------------------
    #dropping 'Ally', MiscFeature, PoolQC because there high percentage of Na values, Uncomment the print line below to see that.
    #print(checkMissingValues(df))
    df = df.drop(['Alley', 'MiscFeature', 'PoolQC', 'Utilities', 'Fence', 'FireplaceQu', 'LotFrontage'], axis = 1)
    
    
    #----------------------------------------------------------------------------------------------------------
    # LotShape
    #checkRange(df, 'LotShape')
    
    
    '''['Reg - 62.2 %', 'IR1 - 34.4 %', 'IR2 - 2.6 %', 'IR3 - 0.7 %'] - the percentage of 
    the values show that it is all Reg and IR1. Hence we would keep all those values that 
    are Reg and IR1 otherwise, we would drop those rows.'''
     
    #df = df.loc[(df.loc[:,'LotShape'] == 'Reg') | (df.loc[:,'LotShape']=='IR1') | (df.loc[:,'LotShape']=='IR2')]
    #df.loc[:,'LotShape'] = df.loc[:,'LotShape'].map(lambda val: 0 if(val=='Reg') else val)
    #df.loc[:,'LotShape'] = df.loc[:,'LotShape'].map(lambda val: 1 if(val=='IR1') else val)
    #df.loc[:,'LotShape'] = df.loc[:,'LotShape'].map(lambda val: 2 if(val=='IR2') else val)
    
    #scatterPlot(df, 'LotShape')
    
    '''Now we would discreditize the data into 0,1 or 2 and find the correlation between Lot Shape
    and SalePrice'''
    
    
    #print(c2(df, 'LotShape'))
    
    #Hence we would drop this too.
    df = colDrop(df, 'LotShape')
    
    
    
    
    #_____________________________________________________________________________________________
    #This is LandContour
    #checkRange(df, 'LandContour')
    #df = df.loc[(df.loc[:,'LandContour'] == 'Lvl') | (df.loc[:,'LandContour'] == 'Bnk') | (df.loc[:,'LandContour'] == 'Low') | (df.loc[:,'LandContour'] == 'HLS')]
    #df.loc[:,'LandContour'] = df.loc[:,'LandContour'].map(lambda val: 0 if(val=='Lvl') else val)
    #df.loc[:,'LandContour'] = df.loc[:,'LandContour'].map(lambda val: 1 if(val=='Bnk') else val)
    #df.loc[:,'LandContour'] = df.loc[:,'LandContour'].map(lambda val: 2 if(val=='Low') else val)
    #df.loc[:,'LandContour'] = df.loc[:,'LandContour'].map(lambda val: 3 if(val=='HLS') else val)
    #histogram(df, 'LandContour')
    #print(c2(df, 'LandContour'))
    '''Since, LandContour and LotShape basically is providing the same information - we would use only
    one of it - the one with the higher correlation with SalePrice, if it exceeds 0.5'''
    df = colDrop(df, 'LandContour')
    
    
    #____________________________________________________________________________________________
    #This is LotConfig
    
    #checkRange(df, 'LotConfig')
    df = df.loc[(df.loc[:,'LotConfig'] == 'Inside') | (df.loc[:,'LotConfig'] == 'FR2') |(df.loc[:,'LotConfig'] == 'Corner') | (df.loc[:,'LotConfig'] == 'CulDsac') ]
    
    df.loc[:,'LotConfig'] = df.loc[:,'LotConfig'].map(lambda val: 0 if(val=='Inside') else val)
    df.loc[:,'LotConfig'] = df.loc[:,'LotConfig'].map(lambda val: 1 if(val=='FR2') else val)
    df.loc[:,'LotConfig'] = df.loc[:,'LotConfig'].map(lambda val: 2 if(val=='Corner') else val)
    df.loc[:,'LotConfig'] = df.loc[:,'LotConfig'].map(lambda val: 3 if(val=='CulDsac') else val)
    
    
    #print(c2(df, 'LotConfig'))
    #Removed Landconfig as well because the correlation is very less
    df = colDrop(df, 'LotConfig')
    
    
    
    #__________________________________________________________________________________________________
    #LandSlope
    
    #checkRange(df, 'LandSlope')
    df = df.loc[(df.loc[:,'LandSlope'] == 'Gtl') | (df.loc[:, 'LandSlope'] == 'Mod')]
    df.loc[:, 'LandSlope'] = df.apply(lambda row: 1 if row.loc['LandSlope'] == 'Gtl' else 0, axis = 1)
    #print(c2(df, 'LandSlope'))
    #It shows a high percentage of Gtl values, therefore - we would just keep those and remove the others
    #and this column - It also shows a very high -negative correlation - therewould we would keep this column
    
    
    #_____________________________________________________________________________________________
    #Condition1 Condition2
    
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 0 if(val=='Artery') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 1 if(val=='Feedr') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 2 if(val=='Norm') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 3 if(val=='RRNn') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 4 if(val=='PosN') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 5 if(val=='PosA') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 6 if(val=='RRNe') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 7 if(val=='RRAe') else val)
    df.loc[:,'Condition1'] = df.loc[:,'Condition1'].map(lambda val: 8 if(val=='RRAn') else val)
    
    #print(c2(df, 'Condition1'))
    
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 0 if(val=='Artery') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 1 if(val=='Feedr') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 2 if(val=='Norm') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 3 if(val=='RRNn') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 4 if(val=='PosN') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 5 if(val=='PosA') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 6 if(val=='RRNe') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 7 if(val=='RRAe') else val)
    df.loc[:,'Condition2'] = df.loc[:,'Condition2'].map(lambda val: 8 if(val=='RRAn') else val)
    
    #print(c2(df, 'Condition2'))
    
    ''' Clearly the correlation between these two coloumns and SalePrice is very low; therefore, we
    would drop these two columns'''
    df = colDrop(df, 'Condition1')
    df = colDrop(df, 'Condition2')
    
    
    #____________________________________________________________________________________________
    #BldgType
    #labels = df.loc[:,'BldgType'].unique()
    #sizes = df.loc[:,"BldgType"].value_counts().values
    #explode=[0.1,0,0,0,0]
    #parcent = 100.*sizes/sizes.sum()
    #labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]
    #print(labels)
    
    '''1 Fam and Duplex are more than 90% of the values for this column therefore we would, 
    binary encode them and see there correlation with the SalePrice'''
    
    a = encode(df.loc[:, 'BldgType'])
    #df.loc[:, '1Fam'] = a.loc[:, '1Fam']
    df.loc[:, 'Duplex'] = a.loc[:, 'Duplex']
    #print(c2(df, '1Fam'))
    #print(c2(df, 'Duplex'))
    
    '''
    1Fam is not highly correlated with the SalePrice; therefore, which can be confirmed
    by there scatter plots, we won't keep that. However, we would keep the Duplex.
    '''
    
    #data = pd.concat([df.loc[:, 'SalePrice'], df.loc[:,'1Fam']], axis=1)
    #data.plot.scatter(x='1Fam', y='SalePrice', ylim=(0,800000));
    
    #data1 = pd.concat([df.loc[:, 'SalePrice'], df.loc[:,'Duplex']], axis=1)
    #data1.plot.scatter(x='Duplex', y='SalePrice', ylim=(0,800000));
    
    '''In both the cases - SalePrice, is barely affected by 1Fam or DuPlex as it is scattered all over the 
    place. Hence we would also not consider BldgType. '''
    df = colDrop(df, 'BldgType')
    
    
    
    
    
    
    
    #____________________________________________________________________________________________
    #RoofMatl
    #checkRange(df, 'RoofMatl')
    
    ''' 99.6 percent of the values are CompShg therefore we would ignore the other values and ignore
    this column'''
    
    df = df.loc[df.loc[:, 'RoofMatl'] == 'CompShg']
    df = colDrop(df, 'RoofMatl')
    
    
    
    
    
    
    #____________________________________________________________________________________________
    #MasVnrType
    #checkRange(df, 'MasVnrType')
    #scatterPlot(df, 'MasVnrArea')
    
    df.loc[:, 'MasVnrType'] == df.loc[:, 'MasVnrType'].fillna('None')
    df.loc[:, 'MasVnrArea'] == df.loc[:, 'MasVnrArea'].fillna(0)
    
    a = df.loc[(df.loc[:,'MasVnrType'] == 'None') & (df.loc[:, 'MasVnrArea'] == 0)]
    
    
    #print(len(a))
    #print(len(df[df.loc[:, 'MasVnrType'] == 'None']))
    #print(len(df[df.loc[:, 'MasVnrArea'] == 0]))
    
    '''This shows a relationship between the two and even from the names we can figure it out that we just need one of it;
    therefore, we would keep the one that is more easier to use (the one that is numeric) and figure out it's corr'''
    df = colDrop(df, 'MasVnrType')
    #print(c2(df, 'MasVnrArea'))
    '''We would keep this because it's correlation with SalePrice is 0.499 which is approximately 0.5 - among the threshold
    we are considering values.'''
    
    
    #____________________________________________________________________________________________
    #ExteriorQuality
    
    #checkRange(df, 'ExterQual')
    '''From this we figure out that Gd, TA, and Ex makes the entire data therefore we would remove the rest'''
    
    df = df.loc[(df.loc[:, 'ExterQual'] == 'Gd') | (df.loc[:, 'ExterQual'] == 'TA') | (df.loc[:, 'ExterQual'] == 'Ex')]
    
    df.loc[:,'ExterQual'] = df.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Gd') else val)
    df.loc[:,'ExterQual'] = df.loc[:,'ExterQual'].map(lambda val: 2 if(val=='TA') else val)
    df.loc[:,'ExterQual'] = df.loc[:,'ExterQual'].map(lambda val: 3 if(val=='Ex') else val)
    
    #histogram(df, 'ExterQual')
    #checkRange(df, 'ExterCond')
    #print(c2(df, 'ExterQual'))
    
    '''Since, it's corr relation is negative and  far from zero, therefore, we would keep this. However, we would not
    use exterior condition because both of them are almost the same thing because in both the cases - TA and Gd make up a large
    portion of the data.'''
    df = colDrop(df, 'ExterCond')
    
    
    
    
    #____________________________________________________________________________________________
    #Foundation
    
    
    #checkRange(df, 'Foundation')
    
    df = df.loc[(df.loc[:, 'Foundation'] == 'PConc') | (df.loc[:, 'Foundation'] == 'CBlock') | (df.loc[:, 'Foundation'] == 'CBlock')]
    df.loc[:,'Foundation'] = df.loc[:,'Foundation'].map(lambda val: 1 if(val=='PConc') else val)
    df.loc[:,'Foundation'] = df.loc[:,'Foundation'].map(lambda val: 2 if(val=='CBlock') else val)
    df.loc[:,'Foundation'] = df.loc[:,'Foundation'].map(lambda val: 3 if(val=='CBlock') else val)
    
    #histogram(df, 'Foundation')
    #scatterPlot(df, 'Foundation')
    
    #print(c2(df, 'Foundation'))
    
    '''The corr is very high negative - therefore we will take this into consideration either.'''
    
    
    
    
    #____________________________________________________________________________________________
    #Basement Features
    
    '''There is high correlation between basement features (BsmtFinType2, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1);
    therefore, we would only use BsmtCond - correlation shown below'''
    
    
    #checkRange(df, 'BsmtQual')
    #checkRange(df, 'BsmtCond')
    
    df.loc[:, 'BsmtQual'] = df.loc[:, 'BsmtQual'].fillna('None')
    df.loc[:, 'BsmtCond'] = df.loc[:, 'BsmtCond'].fillna('None')
    
    
    
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 1 if(val=='Gd') else val)
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 2 if(val=='TA') else val)
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 2 if(val== 'Ex') else val)
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 4 if(val== 'Fa') else val)
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 5 if(val==  'None') else val)
    df.loc[:,'BsmtQual'] = df.loc[:,'BsmtQual'].map(lambda val: 6 if(val== 'Po') else val)
    
    

    
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='Gd') else val)
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='TA') else val)
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 5 if(val==  'None') else val)
    df.loc[:,'BsmtCond'] = df.loc[:,'BsmtCond'].map(lambda val: 6 if(val== 'Po') else val)
    
    #print(df.loc[:, 'BsmtCond'].corr(df.loc[:,'BsmtQual']))
    '''This shows a high correlation  between BsmtCond and BsmtQual - therefore, we would keep BsmtCond'''
    df = colDrop(df, 'BsmtQual')
    
    
    
    
    #____________________________________________________________________________________________
    #Gas
    
    #checkRange(df, 'Heating') #Show 99.6% GasA - therefore keeping only data that is GasA
    df = df.loc[df.loc[:, 'Heating'] == 'GasA']
    df = colDrop(df, 'Heating')
    
    #checkRange(df, 'HeatingQC')
    df.loc[:,'HeatingQC'] = df.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Gd') else val)
    df.loc[:,'HeatingQC'] = df.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='TA') else val)
    df.loc[:,'HeatingQC'] = df.loc[:,'HeatingQC'].map(lambda val: 3 if(val== 'Ex') else val)
    df.loc[:,'HeatingQC'] = df.loc[:,'HeatingQC'].map(lambda val: 4 if(val== 'Fa') else val)
    df.loc[:,'HeatingQC'] = df.loc[:,'HeatingQC'].map(lambda val: 5 if(val== 'Po') else val)
    
    #print(c2(df, 'HeatingQC'))
    #scatterPlot(df, 'HeatingQC')
    
    '''The graph shows that Heating Quality definitely affects the price range towards the upper end.'''
    
    
    
    
    #____________________________________________________________________________________________
    #Central Air 
    #checkRange(df, 'CentralAir')
    df.loc[:,'CentralAir'] = df.loc[:,'CentralAir'].map(lambda val: 0 if(val=='N') else val)
    df.loc[:,'CentralAir'] = df.loc[:,'CentralAir'].map(lambda val: 1 if(val=='Y') else val)
    
    #scatterPlot(df, 'CentralAir')
    #histogram(df, 'CentralAir')
    #print(c2(df, 'CentralAir'))
    df = colDrop(df, 'CentralAir')
    

    
    
    
    #checkRange(df, 'Electrical')
    df = df.loc[(df.loc[:, 'Electrical'] == 'SBrkr') | (df.loc[:, 'Electrical'] == 'FuseF')] #98.8 of the total values make this up
    df.loc[:,'Electrical'] = df.loc[:,'Electrical'].map(lambda val: 0 if(val=='SBrkr') else 1)
    #print(df.loc[:, 'Electrical'])
    #print(c2(df, 'Electrical'))
    '''This is a negative number close to 0; hence, a low corr between Electricity and SalePrice.
    therefore - we would not keep this.'''
    df = colDrop(df, 'Electrical')
    
    
    
    #KitchenQuality
    #checkRange(df, 'KitchenQual')
    ''' 98% of the data is made up of Gd, TA and Ex hence we would only keep, discretize it 
    and then find the correlation between KitcenQual and SalePrice'''
    
    df = df.loc[(df.loc[:, 'KitchenQual'] == 'Gd') | (df.loc[:, 'KitchenQual'] == 'Ex') | (df.loc[:, 'KitchenQual'] == 'TA')]
    df.loc[:,'KitchenQual'] = df.loc[:,'KitchenQual'].map(lambda val: 0 if(val=='Ex') else val)
    df.loc[:,'KitchenQual'] = df.loc[:,'KitchenQual'].map(lambda val: 1 if(val=='TA') else val)
    df.loc[:,'KitchenQual'] = df.loc[:,'KitchenQual'].map(lambda val: 2 if(val=='Gd') else val)
    
    #print(c2(df, 'KitchenQual'))
    
    '''It really has a low negative corr with SalePrice - therefore we would not keep this.'''
    df = colDrop(df, 'KitchenQual')
    
    
    
    #function
    #checkRange(df, 'Functional')
    df = df.loc[(df.loc[:, 'Functional'] == 'Typ') | (df.loc[:, 'Functional'] == 'Min1') | (df.loc[:, 'Functional'] == 'Maj1')]
    df.loc[:,'Functional'] = df.loc[:,'Functional'].map(lambda val: 0 if(val=='Typ') else val)
    df.loc[:,'Functional'] = df.loc[:,'Functional'].map(lambda val: 1 if(val=='Min1') else val)
    df.loc[:,'Functional'] = df.loc[:,'Functional'].map(lambda val: 2 if(val=='Maj1') else val)

    #print(c2(df, 'Functional'))
    
    '''This has negative corr very close to zero; therefore, we would drop it.'''
    df = colDrop(df, 'Functional')
    
    #----------------------------------------------------------------------------------------------------------
    #HouseStyle
    #checkRange(df, 'HouseStyle')
    df = df.loc[(df.loc[:, 'HouseStyle'] == '2Story') | (df.loc[:, 'HouseStyle'] == '1Story') | (df.loc[:, 'HouseStyle'] == '1.5Unf')]
    df.loc[:,'HouseStyle'] = df.loc[:,'HouseStyle'].map(lambda val: 0 if(val=='2Story') else val)
    df.loc[:,'HouseStyle'] = df.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='1Story') else val)
    df.loc[:,'HouseStyle'] = df.loc[:,'HouseStyle'].map(lambda val: 2 if(val=='1.5Unf') else val)

    #print( df.loc[:,'HouseStyle'].dtype)
    '''It is a high negative correlation with SalePrice - therefore we would keep it.'''
    
    
    
    
    
    #----------------------------------------------------------------------------------------------------------
    #RoofStyle
    #checkRange(df, 'RoofStyle')
    '''Data represents a lot like HouseStyle; therefore, assuming a high correlation between
    roofstyle and housestyle - we would drop this as well.'''
    df = colDrop(df, 'RoofStyle')
    
    #_____________________________________________________________
    #Exterior1st
    
    df = colDrop(df, 'Exterior1st')
    df = colDrop(df, 'Exterior2nd')
    
    df = colDrop(df, 'BsmtExposure')
    df = colDrop(df, 'BsmtFinType1')
    df = colDrop(df, 'BsmtFinType2')
    df = colDrop(df, 'GarageType') 
    
    #checkRange(df, 'GarageFinish')
    a = encode(df.loc[:, 'GarageFinish'])
    df = join(df, a)
    
    
    df = colDrop(df,'GarageFinish')
    
    #checkRange(df, 'PavedDrive')
    df = df.loc[(df.loc[:, 'PavedDrive'] == 'Y') | (df.loc[:, 'PavedDrive'] == 'N')]
    df.loc[:,'PavedDrive'] = df.loc[:,'PavedDrive'].map(lambda val: 0 if(val=='N') else 1)
    #print(c2(df, 'PavedDrive'))
    '''Very low correlation - hence we would drop this.'''
    df = colDrop(df, 'PavedDrive')
    df = colDrop(df, 'GarageCond')
    df = colDrop(df, 'GarageQual')
    
    
    #checkRange(df, 'SaleType')
    df = df.loc[(df.loc[:, 'SaleType'] == 'WD') | (df.loc[:, 'SaleType'] == 'New') | (df.loc[:, 'SaleType'] == 'COD')]
    df.loc[:,'SaleType'] = df.loc[:,'SaleType'].map(lambda val: 0 if(val=='WD') else val)
    df.loc[:,'SaleType'] = df.loc[:,'SaleType'].map(lambda val: 1 if(val=='New') else val)
    df.loc[:,'SaleType'] = df.loc[:,'SaleType'].map(lambda val: 2 if(val=='COD') else val)
    df = colDrop(df, 'SaleType')
    
    #checkRange(df, 'SaleCondition')
    df = df.loc[(df.loc[:, 'SaleCondition'] == 'Normal') | (df.loc[:, 'SaleCondition'] == 'Partial') | (df.loc[:, 'SaleCondition'] == 'Abnormal')]
    df.loc[:,'SaleCondition'] = df.loc[:,'SaleCondition'].map(lambda val: 0 if(val=='Normal') else val)
    df.loc[:,'SaleCondition'] = df.loc[:,'SaleCondition'].map(lambda val: 1 if(val=='Partial') else val)
    df.loc[:,'SaleCondition'] = df.loc[:,'SaleCondition'].map(lambda val: 2 if(val=='Abnormal') else val)

    #print(c2(df, 'SaleCondition'))
    
    
    #_handling the NA values 
    df.loc[:,'MasVnrArea'] = df.loc[:,'MasVnrArea'].fillna(0)
    #df.loc[:,'BsmtExposure'] = df.loc[:,'BsmtExposure'].fillna(df.loc[:,'BsmtExposure'].mode()[0])
    #df.loc[:,'BsmtFinType1'] = df.loc[:,'BsmtFinType1'].fillna(df.loc[:,'BsmtFinType1'].mode()[0])
    #df.loc[:,'BsmtFinType2'] = df.loc[:,'BsmtFinType2'].fillna(df.loc[:,'BsmtFinType2'].mode()[0])
    #df.loc[:,'GarageType'] = df.loc[:,'GarageType'].fillna(df.loc[:,'GarageType'].mode()[0])
    df.loc[:,'GarageYrBlt'] = df.loc[:,'GarageYrBlt'].fillna(df.loc[:,'GarageYrBlt'].mean())
    #df.loc[:,'GarageFinish'] = df.loc[:,'GarageFinish'].fillna(df.loc[:,'GarageFinish'].mode()[0])
    #df.loc[:,'GarageQual'] = df.loc[:,'GarageQual'].fillna(df.loc[:,'GarageQual'].mode()[0])
    #df.loc[:,'GarageCond'] = df.loc[:,'GarageCond'].fillna(df.loc[:,'GarageCond'].mode()[0])
    
    
    #corr=df.corr()["SalePrice"]
    #corr[np.argsort(corr, axis=0)[::-1]]
    #print(corr)
    
    '''Cols that we would keep because the corr relation with SalePrice are relatively higher
    
    'LotArea', 'HouseStyle', 'OverallQual', 'OverallCond', 'OverallCond',
                  'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'Foundation', 'BsmtCond',
                  'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'KitchenAbvGr',
                  'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'Unf'
     '''  
      
    df = colDrop(df, 'MSSubClass')
    df = colDrop(df, 'Fireplaces')
    df = colDrop(df, 'LandSlope')
    df = colDrop(df, 'BsmtFinSF1')
    df = colDrop(df, 'BsmtFinSF2')
    df = colDrop(df, 'BsmtUnfSF')
    df = colDrop(df, 'HeatingQC')
    df = colDrop(df, '2ndFlrSF')
    df = colDrop(df, 'LowQualFinSF')
    df = colDrop(df, 'BsmtHalfBath')
    df = colDrop(df, 'FullBath')
    df = colDrop(df, 'HalfBath')
    df = colDrop(df, 'BedroomAbvGr')
    df = colDrop(df, 'KitchenAbvGr')
    df = colDrop(df, 'TotRmsAbvGrd')
    df = colDrop(df, 'GarageCars')
    df = colDrop(df, 'WoodDeckSF')
    df = colDrop(df, 'OpenPorchSF')
    df = colDrop(df, '3SsnPorch')
    df = colDrop(df, 'ScreenPorch')
    df = colDrop(df, 'PoolArea')
    df = colDrop(df, 'MiscVal')
    df = colDrop(df, 'MoSold')
    df = colDrop(df, 'YrSold')
    df = colDrop(df, 'Fin')
    df = colDrop(df, 'RFn')
    df = colDrop(df, 'EnclosedPorch') 
    
     
    
    
    return df 
    
    
    
    
    
   
    
    

    
    
    
    
    
            
        
    
    
    
    
    
    



    
    
 






    
def standardize(df, ls):
    
    df.loc[:, ls] =  (df.loc[:, ls] - df.loc[:, ls].mean())
    df.loc[:, ls] = (df.loc[:, ls])/df.loc[:, ls].std()








    
    

def test10():
    df  = dataProcessing()
    inputCols = ['Id', 'LotArea', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                  'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'Foundation', 'BsmtCond',
                  'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath',
                  'GarageYrBlt', 'GarageArea', 'SaleCondition', 'Duplex', 'Unf']
    
    outputCol = ['SalePrice']
    
    #print(df.dtypes)
    
    
    
    standardize(df, inputCols + outputCol)
    
    
    
    df1 = df.loc[:,inputCols]
    outputSeries = df.loc[:,outputCol]
    
    alg = GradientBoostingRegressor()
    alg.fit(df1, outputSeries)
    cvScores = model_selection.cross_val_score(alg, df1, outputSeries ,cv=10, scoring='r2')
    print(cvScores.mean())

    
    
    '''
    alg =  LogisticRegression()

    df1 = df.loc[:, inputCols]
    df2 = df.loc[:, outputCol]
   
    standardize(df1, inputCols)
    standardize(df2, outputCol)
    
    #FROM:https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    #Explanation: casts flaot to int types. As an error came forward
    df2=df2.astype('int')
    alg.fit(df1,df2.values.ravel())
    #END: https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
    
   
    cvScores = model_selection.cross_val_score(alg, df1, df2 ,cv=10, scoring='accuracy')
    orginal_cvScore_mean = cvScores.mean()
    print(orginal_cvScore_mean)

    '''
    
    
    
    
    
    
    
    
    
    
    



    
    
    
    
    