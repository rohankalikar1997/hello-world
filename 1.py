import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from scipy import stats



cutlet = pd.read_csv("D:\\360Digitmg\\Assignments\\hypothesis\\Datasets_HT\\Cutlets.csv")
cutlet.columns
cutlet.columns = cutlet.columns.str.replace(' ', '_')
cutlet.columns

#cutlet = cutlet.rename({'YearsExperience_(grams)': 'YearsExperience'}, axis=1)


cutlet.head()
cutlet.info()



#data preparation
cutlet.isna().sum() #There are missing values
cutlet = cutlet.dropna()
cutlet.isna().sum() #There are no missing values now


cutlet.dtypes #obtain the datatypes
cutlet.shape # shape
cutlet.columns #view all the columns

cutlet.describe()

#check for outliers
cutlet.boxplot() #None present

#outlier treatment
cutlet=cutlet.mask(cutlet.sub(cutlet.mean()).div(cutlet.std()).abs().gt(2)) #mask the outliers
cutlet.isna().sum()
cutlet=cutlet.fillna(cutlet.mean()) #replace the outliers with the means
cutlet.isna().sum()


cutlet.boxplot()
cutlet.describe()
cutlet.shape #verify the dimensions again

## EDA ##
#First moment business decision
cutlet.mean() #mean
cutlet.median() #median
cutlet.mode() #mode

#Second moment business decision
cutlet.var() #variance
cutlet.std() #std deviation


# Third moment business decision
cutlet.skew() #skeweness

# Fourth moment business decision 
cutlet.kurt() #kurtosis

#Graphical representation

#histogram
cutlet.hist()

#boxplot
cutlet.boxplot()



#Bivariate analysis
corr= cutlet.corr(method= 'spearman') #define the correlation between the variables
plt.figure(figsize=(15,15))

#heatmap
sns.heatmap(corr, vmax = 8,linewidth=0.01, square=True, annot=True, cmap='RdBu', linecolor='black')



#####################################################################
cutlet.columns
plt.bar(height = cutlet.Unit_A, x = np.arange(1, 36, 1))
plt.hist(cutlet.Unit_A) #histogram
plt.boxplot(cutlet.Unit_A) #boxplot

plt.bar(height = cutlet.Unit_B, x = np.arange(1, 36, 1))
plt.hist(cutlet.Unit_B) #histogram
plt.boxplot(cutlet.Unit_B) #boxplot

# Scatter plot
plt.scatter(x = cutlet['Unit_B'], y = cutlet['Unit_A'], color = 'green') 

##################################################################

#Hypothesis testing
cutlet.columns

#Normality test

stats.shapiro(cutlet.Unit_A) # Shapiro Test
stats.shapiro(cutlet.Unit_B) # Shapiro Test
#p high, Null fly

#We failed to reject Ho, Ho is accepted,
#Data is normal

#Variance test
scipy.stats.levene(cutlet.Unit_A, cutlet.Unit_B)
#We failed to reject Ho, so we accept Ho
#Variances are equal


#2 sample T test
scipy.stats.ttest_ind(cutlet.Unit_A, cutlet.Unit_B)

# we failed to reject Ho, we accept Ho.
# Thus the means are equal and not significantly different
