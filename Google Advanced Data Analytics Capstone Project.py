#!/usr/bin/env python
# coding: utf-8

# #### Initially, we need to import packages & load datasets

# In[4]:


# Importing packages


# 1.Data manipulation:
import numpy as np
import pandas as pd

# Installing xgboost:
#forge xgboost

import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
# 2.Data visualization:
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb




# 3.For displaying all of the columns in dataframes:
pd.set_option('display.max_columns', None)

# 4.For data modelling:
#from xgboost import XGBClassifier, XGBRegressor, plot_importance


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 5.For metircs calculations & other helpful functions:
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# 6.For Saving models: 
import pickle


# Now, we are going to load the datasets required for analysis

# In[5]:


# Loading dataset into the dataframe:
df= pd.read_csv("/Users/shivamprabhatjha/Downloads/HR_capstone_dataset.csv")
#df = open(r"HR_capstone_dataset.csv","r+")


# To display first few rows of the dataframe:
df.head()


# # Data Exploration
# * Understanding the variables
# * Removing missing, redundant data & any outrliers.

# #### Initially, gathering the basic information about the dataset
# 

# In[6]:


# To gather basic info from the dataset
df.info()


# #### Gathering descriptive statistics about the dataset

# In[7]:


# Gathering the descriptive information about the dataset
df.describe()


# #### Renaming of columns which have misspelled words or some errors

# In[8]:


# Renaming columns as needed:
df= df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Now displaying the new column names:
df.columns


# #### To check for missing values in the dataset

# In[9]:


# Missing values
df.isna().sum()


# #### To check for duplicate data

# In[10]:


# Duplicated data
df.duplicated().sum()


# #### Inspecting the duplicated rows to check the data

# In[11]:


# Inspection of duplicated rows:
df[df.duplicated()].head()


# #### We need to remove all the duplicate rows from the dataset and save the remaining rows in a new dataframe.
# 

# In[12]:


df1 = df.drop_duplicates(keep='first')

#displaying the remaining saved dataframe for QC

df1.head()


# #### Checking for outliers for QC purposes

# In[13]:


# Creating boxplots to visualize the tenure of the employees and check for any outliers.
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()


# The boxplot above shows that the dataset has outliers in the "tenure" variable.

# #### Now to investigate the outliers present in the dataset, we check the row containing the outliers using the 'tenure' column.

# In[14]:


#To find the number of rows containing the outliers:

# Computing the 25th percentile value in 'tenure':
percen_25 = df1['tenure'].quantile(0.25)

# Computing the 75th percentile value in 'tenure':
percen_75 = df1['tenure'].quantile(0.75)

# Computing the inter-quartile range:
inter_quar_range = percen_75 - percen_25

# Now, defining the upper and lower limit for the data apart from outliers:
upper_limit = percen_75 + 1.5*inter_quar_range
lower_limit = percen_25 - 1.5*inter_quar_range

#Printing the values for the upper and lower limit for the dataset:
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Now, we create a subset of the dataset containing the outliers:
#for i in df1 :
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
 
 #outliers2= df1(df1['tenure']< lower_limit)
 
  #Outliers = outliers1 + outliers2
# Now counting the number of row in the dataset which contains the outliers:
count = len(outliers)
print(count)





# In the up-coming stage while selecting the model type, we will need to consider the possibility of removing these outliers as they might effect the final results in a much more drastic way.

# # Step 2: Data Explorations (EDA)
# 

# Now we drill down into understanding the number of employees who have left the organization and what percentage id that of the overall employee number.

# In[15]:


# Finding number of people who left vs stayed:
print(df1['left'].value_counts())
print()


# Finding the percentage of people who left vs stayed:
print(df1['left'].value_counts(normalize=True))




# # Data Visualizations

# You could start by creating a stacked boxplot showing average_monthly_hours distributions for number_project, comparing the distributions of employees who stayed versus those who left.
# 
# Box plots are very useful in visualizing distributions within data, but they can be deceiving without the context of how big the sample sizes that they represent are. So, you could also plot a stacked histogram to visualize the distribution of number_project for those who stayed and those who left.

# In[16]:


# Code for plots:

# Set figure & axes:
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Boxplot showing the average_monthly_hours, number_of_projects handled by each employee. Comparison between the ones which are still present in the organization and the ones who left.
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Creating histogram to compare the number of projects handled by the employees who are still working and the ones who left the organization:
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()


# So its a general notion that if an employee is working on more projects than he/she is going to have more no.of average hours per month. But there are some of the points which stands out:
# 
# 1. So basically we have 2 types of people leaving the organization:
#     A) The ones with very less avg_monthly_hours and working on very less projects.
#     B) The ones with very high avg_monthly_hours and working on large number of projects.
#    People in group A are the ones who already have either resigned and serving there notice periods or are the ones who are going to leave the organization in sometime.People of group B are the ones who are some of the large contributers to the projects they are involved in. These people are the ones who have high avg_working_hours then the rest of the people
#     
# 2. Everyone with seven projects left the company, and the interquartile ranges of this group and those who left with six projects was ~255–295 hours/month—much more than any other group.
# 
# 3. The optimal number of projects for employees to work on seems to be 3–4. The ratio of left/stayed is very small for these cohorts.
# 
# 4. If you assume a work week of 40 hours and two weeks of vacation per year, then the average number of working hours per month of employees working Monday–Friday = 50 weeks * 40 hours per week / 12 months = 166.67 hours per month. This means that, aside from the employees who worked on two projects, every group—even those who didn't leave the company—worked considerably more hours than this. It seems that employees here are overworked.

# The number of employees handling 7 projects together have all left the job.

# In[17]:


# no.of employees who were working on 7 projects, out of them how many of them stayed/left:
df1[df1['number_project']==7]['left'].value_counts()


# Now, comparing the average monthly hours of the employees and there satisfaction level with the help of scatterplot.

# In[18]:


#Creating scatterplot consisting of average_monthly_hours and the satisfaction level of the employees who stayed and the ones who left:
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# The scatterplot above shows that there was a sizeable group of employees who worked ~240&ndash;315 hours per month. 315 hours per month is over 75 hours per week for a whole year. It's likely this is related to their satisfaction levels being close to zero. 
# 
# The plot also shows another group of people who left, those who had more normal working hours. Even so, their satisfaction was only around 0.4. It's difficult to speculate about why they might have left. It's possible they felt pressured to work more, considering so many of their peers worked more. And that pressure could have lowered their satisfaction levels. 
# 
# Finally, there is a group who worked ~210&ndash;280 hours per month, and they had satisfaction levels ranging ~0.7&ndash;0.9. 
# 
# Note the strange shape of the distributions here. This is indicative of data manipulation or synthetic data. 

# In[19]:


# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Creating boxplots to show the relationship between the satisfaction level of the employees and the tenure of the employees:
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Creating histograms to show the relationship between the satisfaction level of the employees and the tenure of the employees:
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();


# There are many observations you could make from this plot.
# - Employees who left fall into two general categories: dissatisfied employees with shorter tenures and very satisfied employees with medium-length tenures.
# - Four-year employees who left seem to have an unusually low satisfaction level. It's worth investigating changes to company policy that might have affected people specifically at the four-year mark, if possible. 
# - The longest-tenured employees didn't leave. Their satisfaction levels aligned with those of newer employees who stayed. 
# - The histogram shows that there are relatively few longer-tenured employees. It's possible that they're the higher-ranking, higher-paid employees.

# Next, we will calculate the mean and the median of the satisfaction scores of the employees who are still in the working and the ones who left

# In[20]:


# Calculate mean and median satisfaction scores of employees who left and those who stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# As expected, the mean and median satisfaction scores of employees who left are lower than those of employees who stayed. Interestingly, among employees who stayed, the mean satisfaction score appears to be slightly below the median score. This indicates that satisfaction levels among those who stayed might be skewed to the left. 
# 

# Next, examining the salary levels for different tenures

# In[21]:


# Creating plots to compare the salary levels of employees with different tenures

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# For Short-term employees:
tenure_short = df1[df1['tenure'] < 7]

# For Long-term employees:
tenure_long = df1[df1['tenure'] > 6]

# Plotting histogram for short-tenured employees:
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14');

# Plotting histogram for long-tenured employees: 
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');


# The plots above show that long-tenured employees were not disproportionately comprised of higher-paid employees.

# We can also check for any correlation between the long working hours & the evaluation scores of the employees
# 

# In[22]:


# Creating scatterplot to understand the correlation between avg. long working hours and the last evaluation.

plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# The following observations can be made from the scatterplot above:
# - The scatterplot indicates two groups of employees who left: overworked employees who performed very well and employees who worked slightly under the nominal monthly average of 166.67 hours with lower evaluation scores. 
# - There seems to be a correlation between hours worked and evaluation score. 
# - There isn't a high percentage of employees in the upper left quadrant of this plot; but working long hours doesn't guarantee a good evaluation score.
# - Most of the employees in this company work well over 167 hours per month.
# 
# 

# Now we can check whether the employees who are workign for really long hours, have they got any promotions in the last 5 years or so.
# 

# In[23]:


#Creating scatterplot to correlate between avg. long working hours and the no.of promotions in the last 5 years
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');


# The plot above shows the following:
# - very few employees who were promoted in the last five years left
# - very few employees who worked the most hours were promoted
# - all of the employees who left were working the longest hours  
# 

# Distribution of employees across the departments who left.

# In[24]:


# Display counts for each department
df1["department"].value_counts()


# In[25]:


# Creating histogram to compare the no.of employees who stayed or left the company on department basis.
plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1, 
             hue_order=[0,1], multiple='dodge', shrink=.5)
plt.xticks(rotation='vertical')
plt.title('Counts of stayed/left by department', fontsize=14);


# In[33]:


df.info()


# There doesn't seem to be any department that differs significantly in its proportion of employees who left to those who stayed. 

# Now lastly, we can check for strong correlations between all the variables

# In[34]:


# Plot a correlation heatmap
plt.figure(figsize=(16, 9))
subset_df = df[['satisfaction_level','last_evaluation','number_project','average_monthly_hours','tenure','work_accident','left','promotion_last_5years','department','salary']]
heatmap = sns.heatmap(subset_df.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);


# The correlation heatmap confirms that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level.

# ### Insights
# 
# 
# It seems that inadequate management is the reason why workers are quitting the company. Longer workdays, a heavy workload, and usually worse job satisfaction are all associated with leaving. Working hard hours and not getting promoted or receiving excellent review ratings might be discouraging. This company employs a sizable number of people that are most likely burnt out. Additionally, it seems that workers who have been with the organization for more than six years typically stay on.

# # Step 3: Model building
# 

# - Model selection to find a model which can predict outcome while working with 2 or more variables
# - Check model assumptions
# - Need to evaluate the model and select the one with the right set of precision and F1 score value.

# ## Model 1: Logistic Regression Modelling
# 

# We can use binomial logistic regression as it suits the task on hand beacause it involves binary classification.
# We need to encode the non-numeric values like departments & salary.
# - department is a categorical variable which can be utlised by dummying it for modelling
# - salary is ordinal, so rather than converting the variable into categorical variable, we convert it into numerical classification levels from 0-2.
# 
# 

# In[30]:


# Copying the dataframe:
df_enc = df1.copy()

# Now encoding the salary column as numerical category:
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

#Now dummy encoding the departments column:
df_enc = pd.get_dummies(df_enc, drop_first=False)

#Now the new dataframe:
df_enc.head()


# Now, we create heatmap to find the correlations between the variables.

# In[31]:


plt.figure(figsize=(8, 6))
sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()


# Now we will create bar plots to compare the number of employees who are still working to the ones who left on thee basis of there department.

# In[32]:


# Stacked bar plots to compare the number of employees who stayed/left based on there department.
# In the legend, 0(purple color) represents employees who did not leave, 1(red color) represents employees who left.
pd.crosstab(df1['department'], df1['left']).plot(kind ='bar',color='mr')
plt.title('Counts of employees who left versus stayed across department')
plt.ylabel('Employee count')
plt.xlabel('Department')
plt.show()


# Now we take the next step to remove the outliers from the 'tenure' column, since logistic regression is sensitive to outliers.

# In[33]:


# Select rows without outliers in `tenure` and save resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]

# Display first few rows of new dataframe
df_logreg.head()


# Now we want to isolate the outcome variable so that we can get tehe outcome we need

# In[34]:


# Isolate the outcome variable
y = df_logreg['left']

# Display first few rows of the outcome variable
y.head() 


# Now selecting the features we want to use in our model. It will help us to predict the outcome variable.

# In[35]:


# Select the features you want to use in your model
X = df_logreg.drop('left', axis=1)

# Display the first few rows of the selected features 
X.head()


# Now, we are going to split the dataset into training and test dataset. Also stratification of value rows will be required on the basis of the values of y, as there is class imbalance present in the data.

# In[36]:


# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


# Constructing a logistic regression model & fitting it to the training dataset.

# In[37]:


# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# Test the logistic regression model,use the logistic regression model to get the outputs on the test set.

# In[38]:


# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)


# Now creating a confusion matrix which can be used to visualize the results of the logistic regression model.

# In[39]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, 
                                  display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot(values_format='')

# Display plot
plt.show()


# The upper-left quadrant displays the number of true negatives.
# The upper-right quadrant displays the number of false positives.
# The bottom-left quadrant displays the number of false negatives.
# The bottom-right quadrant displays the number of true positives.
# 
# True negatives: The number of people who did not leave that the model accurately predicted did not leave.
# 
# False positives: The number of people who did not leave the model inaccurately predicted as leaving.
# 
# False negatives: The number of people who left that the model inaccurately predicted did not leave
# 
# True positives: The number of people who left the model accurately predicted as leaving
# 

# A pefect model in reality, would yield all true positives and true negatives, & no false negatvies and false positives.
# Now, we create a classification report which helps us to include precision, recall, f1-score & accuract metrics to evaluate the performance of the logistic regression model.
# we first check the class balance of the data to make sure about the % of stayed vs. left data

# In[40]:


df_logreg['left'].value_counts(normalize=True)


# There is an approximately 83%-17% split. So the data is not perfectly balanced, but it is not too imbalanced. If it was more severely imbalanced, you might want to resample the data to make it more balanced. In this case, you can use this data without modifying the class balance and continue evaluating the model.

# In[41]:


#Creating classification report for the logistic regression model.
target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))


# The classification report above shows that the logistic regression model achieved a precision of 79%, recall of 82%, f1-score of 80% (all weighted averages), and accuracy of 82%. However, if it's most important to predict employees who leave, then the scores are significantly lower.

# ## Model 2: Tree based modelling

# This approach covers both Random forest and Decision tree implementation

# Isolating the outcome variable.

# In[42]:


# Isolate the outcome variable
y = df_enc['left']

# Display the first few rows of `y`
y.head()


# Selecting the features

# In[43]:


# Select the features
X = df_enc.drop('left', axis=1)

# Display the first few rows of `X`
X.head()


# Splitting the data into training, validating & the test sets

# In[44]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# ### 1. Decision Tree Modelling

# Now, creating a decision tree model and set up a cross-validated grid-search to search exhaustively for the best parameters.

# In[45]:


# Instantiating the model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiating GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# Fitting the decision tree to the training data.

# In[46]:


get_ipython().run_cell_magic('time', '', 'tree1.fit(X_train, y_train)\n')


# Optimal values for the decision tree parameters

# In[47]:


# Check best parameters
tree1.best_params_


# This is a strong AUC score, which shows that this model can predict employees who will leave very well.

# In[48]:


# Check best AUC score on CV
tree1.best_score_


# insights from the above best score.

# Writing a function to extract all the scores from the grid search.

# In[49]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                  }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                        })
  
    return table


# Using the function just defined to get all the scores from the grid search

# In[50]:


# Get all CV scores
tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
tree1_cv_results


# All of these scores from the decision tree model are strong indicators of good model performance. 
# 
# Recall that decision trees can be vulnerable to overfitting, and random forests avoid overfitting by incorporating multiple trees to make predictions. You could construct a random forest model next.

# ### 2. Random forest modelling

# Now, creating a random forest model and set up a cross-validated grid-search to search exhaustively for the best parameters.

# In[51]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# Fitting the random forest model to the training dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf1.fit(X_train, y_train) # --> Wall time: ~10min\n')


# Specifying path to where u want to save your model

# In[ ]:


# Define a path to the folder where you want to save the model
path = '/home/jovyan/work/'


# Defining funtions to pickle the model and read the data into the model

# In[ ]:


def write_pickle(path, model_object, save_as:str):
    '''
    In: 
        path:         path of folder where you want to save the pickle
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open(path + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)


# In[ ]:


def read_pickle(path, saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model


# Use the functions defined above to save the model in a pickle file and then read it in.

# In[ ]:


# Write pickle
write_pickle(path, rf1, 'hr_rf1')


# In[ ]:


# Read pickle
rf1 = read_pickle(path, 'hr_rf1')


# 
# Identify the best AUC score achieved by the random forest model on the training set.

# In[ ]:


# Check best AUC score on CV
rf1.best_score_


# Identify the optimal values for the parameters of the random forest model.

# In[ ]:


# Check best params
rf1.best_params_


# 
# Collect the evaluation scores on the training set for the decision tree and random forest models.

# In[ ]:


# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)


# The evaluation scores of the random forest model are better than those of the decision tree model, with the exception of recall (the recall score of the random forest model is approximately 0.001 lower, which is a negligible amount). This indicates that the random forest model mostly outperforms the decision tree model.
# 
# Next, you can evaluate the final model on the test set.

# Define a function that gets all the scores from a model's predictions.

# In[ ]:


def get_scores(model_name:str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In: 
        model_name (string):  How you want your model to be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision], 
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                         })
  
    return table


# Now use the best performing model to predict on the test set.

# In[ ]:


# Get predictions on test data
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
rf1_test_scores


# The test scores are very similar to the validation scores, which is good. This appears to be a strong model. Since this test set was only used for this model, you can be more confident that your model's performance on this data is representative of how it will perform on new, unseeen data.

# #### Feature Engineering 
# 
# The high evaluation scores may raise doubts in your mind. There's a potential that some data is being leaked. When you utilize data that shouldn't be used for training—either because it shows up in test data or because it doesn't match the data you would anticipate having when the model is actually deployed—you are engaging in data leaking. When a model is trained with leaked data, it may produce an inflated score that is not repeated in real life.
# 
# In this instance, it's possible that not all of the company's employees' satisfaction scores will be available. Another possibility is that there is some data leakage occurring in the average_monthly_hours column. If workers have already made the decision to leave or have been recognized by management as resigning,

# In the next round we drop the satisafaction level data from our dataset available and we go for a new variable.
# 

# In[ ]:


# Drop `satisfaction_level` and save resulting dataframe in new variable
df2 = df_enc.drop('satisfaction_level', axis=1)

# Display first few rows of new dataframe
df2.head()


# In[ ]:


# Create `overworked` column. For now, it's identical to average monthly hours.
df2['overworked'] = df2['average_monthly_hours']

# Inspect max and min average monthly hours values
print('Max hours:', df2['overworked'].max())
print('Min hours:', df2['overworked'].min())


# 166.67 is approximately the average number of monthly hours for someone who works 50 weeks per year, 5 days per week, 8 hours per day. 
# 

# We are progressing towards making the 'overworked' column as binary.
# df3['overworked'] > 175 creates a series of booleans, consisting of True for every value > 175 and False for every values ≤ 175.astype(int) converts all True to 1 and all False to 0.
# 
# You could define being overworked as working more than 175 hours per month on average.
# 

# In[ ]:


# Define `overworked` as working > 175 hrs/week
df2['overworked'] = (df2['overworked'] > 175).astype(int)

# Display first few rows of new column
df2['overworked'].head()


# Dropping the average_monthly_hours column from the dataset

# In[ ]:


# Drop the `average_monthly_hours` column
df2 = df2.drop('average_monthly_hours', axis=1)

# Display first few rows of resulting dataframe
df2.head()


# Again, isolate the features and target variables

# In[ ]:


# Isolate the outcome variable
y = df2['left']

# Select the features
X = df2.drop('left', axis=1)


# Split the data into training and testing sets.

# In[ ]:


# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# #### Decision Tree Modelling (Round 2)

# In[ ]:


# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tree2.fit(X_train, y_train)\n')


# In[ ]:


# Check best params
tree2.best_params_


# In[ ]:


# Check best AUC score on CV
tree2.best_score_


# This model performs very well, even without satisfaction levels and detailed hours worked data.

# Next, check the other scores.

# In[ ]:


# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)


# Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.

# #### Random forest modelling (Round 2)

# In[ ]:


# Instantiate model
rf = RandomForestClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }  

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf2.fit(X_train, y_train) # --> Wall time: 7min 5s\n')


# In[ ]:


# Write pickle
write_pickle(path, rf2, 'hr_rf2')


# In[ ]:


# Read in pickle
rf2 = read_pickle(path, 'hr_rf2')


# In[ ]:


# Check best params
rf2.best_params_


# In[ ]:


# Check best AUC score on CV
rf2.best_score_


# Insights on the above score table

# In[ ]:


# Get all CV scores
rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)


# Again, the scores dropped slightly, but the random forest performs better than the decision tree if using AUC as the deciding metric.
# 
# Score the champion model on the test set now.

# In[ ]:


# Get predictions on test data
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
rf2_test_scores


# This seems to be a stable, well-performing final model. 

# Plot a confusion matrix to visualize how well it predicts on the test set.

# In[39]:


# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)
disp.plot(values_format='');


# The model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. But this is still a strong model.
# 
# For exploratory purpose, you might want to inspect the splits of the decision tree model and the most important features in the random forest model. 

# #### Decision tree splits

# In[ ]:


# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()


# #### Decision tree feature importance

# In[ ]:


#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, 
                                 columns=['gini_importance'], 
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
tree2_importances


# You can then create a barplot to visualize the decision tree feature importances.

# In[ ]:


sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()


# The barplot above shows that in this decision tree model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`.

# #### Random forest feature importance

# Now, plot the feature importances for the random forest model.

# In[ ]:


# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features 
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()


# The plot above shows that in this random forest model, `last_evaluation`, `number_project`, `tenure`, and `overworked` have the highest importance, in that order. These variables are most helpful in predicting the outcome variable, `left`, and they are the same as the ones used by the decision tree model.

# # Step 4: Results & Evaluation
# - Interpreting the selected model.
# - Evaluating the model performance using metrics.
# - Prepare results, visualizations, and actionable steps to share with stakeholders.

# ### Summary of model results
# 
# **Logistic Regression**
# 
# The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.
# 
# **Tree-based Machine Learning**
# 
# After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model. 

# ### Conclusion, Recommendations, Next Steps
# 
# The models and the feature importances extracted from the models confirm that employees at the company are overworked. 
# 
# To retain employees, the following recommendations could be presented to the stakeholders:
# 
# * Cap the number of projects that employees can work on.
# * Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied. 
# * Either reward employees for working longer hours, or don't require them to do so. 
# * If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear. 
# * Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts. 
# * High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort. 
# 
# **Next Steps**
# 
# It may be justified to still have some concern about data leakage. It could be prudent to consider how predictions change when `last_evaluation` is removed from the data. It's possible that evaluations aren't performed very frequently, in which case it would be useful to be able to predict employee retention without this feature. It's also possible that the evaluation score determines whether an employee leaves or stays, in which case it could be useful to pivot and try to predict performance score. The same could be said for satisfaction score. 
# 
# For another project, you could try building a K-means model on this data and analyzing the clusters. This may yield valuable insight. 

# In[ ]:




