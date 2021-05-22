#!/usr/bin/env python
# coding: utf-8

# # Vinho Verde - Wine Dataset Predicting Wine Type 
# ### My dataframe - Wines
# #### Machine Learning - Predicting Wine Type (Red or White)

# In[11]:


# Set up my wine dataset - My DataFrame

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[3]:


RW_df = pd.read_csv('winequality-red.csv', delimiter = ';')
WW_df = pd.read_csv('winequality-white.csv', delimiter = ';')


# In[4]:


WW_df['type'] = 'White Wine'
RW_df['type'] = 'Red Wine'


# In[5]:


df_wines = pd.concat([WW_df, RW_df])


# In[6]:


def label(x):
    if (x == 8) or (x == 9) or (x == 7)  :
        return 'High'
    
    elif x in [5, 6]:
        return 'Medium'
    else:
        return 'Low'


# In[7]:


df_wines['quality_label'] = df_wines['quality'].apply(lambda x: label(x))


# In[8]:


df_wines['type'].value_counts()


# #### Splitting the Data following the LMS code - Basic Preprocessing 

# In[49]:


# wtp = wine type prediction 

wtp_features = df_wines.iloc[:,:-3] # X

wtp_feature_names = wtp_features.columns

wtp_class_labels = np.array(df_wines['type']) # y

wtp_train_X, wtp_test_X, wtp_train_y, wtp_test_y = train_test_split(wtp_features,    # Splitting Train and Test
wtp_class_labels, test_size=0.3, random_state=42)

print(Counter(wtp_train_y), Counter(wtp_test_y))
print('Features:', list(wtp_feature_names))


# In[13]:


# ---


# ### Preprocessing

# In[20]:


# Define the scaler
   
wtp_ss = StandardScaler().fit(wtp_train_X)

# Scale the train set

wtp_train_SX = wtp_ss.transform(wtp_train_X)

# Scale the test set

wtp_test_SX = wtp_ss.transform(wtp_test_X)


from sklearn.preprocessing import StandardScaler, LabelEncoder


le = LabelEncoder() 

df_wines['type'] = le.fit_transform(df_wines['type'])

df_wines['type'].value_counts()


# ### Modelling

# #### 1-) Logistic Regression - Using the LMS (Code Academy Berlin) Codes

# In[21]:


from sklearn.linear_model import LogisticRegression

wtp_lr = LogisticRegression()

wtp_lr.fit(wtp_train_SX, wtp_train_y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)


# In[22]:


# The model is ready.


# In[23]:


from sklearn.metrics import classification_report

wtp_lr_predictions = wtp_lr.predict(wtp_test_SX)

print(classification_report(wtp_test_y,wtp_lr_predictions, target_names=['Red', 'White']))


# In[30]:


# Cohen's Kappa 


# In[31]:


from sklearn.metrics import cohen_kappa_score


# In[32]:


cohen_kappa_score(wtp_test_y,wtp_lr_predictions)


# In[39]:


# ---


# #### 2-) Decision Tree

# In[40]:


from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier


# In[41]:


# train the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report


# In[43]:


wtp_features = df_wines.iloc[:,:-3]

wtp_class_labels = np.array(df_wines['type'])

wtp_label_names = ['Red', 'White']

wtp_feature_names = list(wtp_features.columns)

wtp_train_X, wqp_test_X, wtp_train_y, wtp_test_y = train_test_split(wtp_features,
wtp_class_labels, test_size=0.3, random_state=42)

print(Counter(wtp_train_y), Counter(wtp_test_y))

print('Features:', wtp_feature_names)


# In[ ]:





# In[48]:


wtp_dt = DecisionTreeClassifier()


# In[ ]:





# In[51]:


# Fitting the model

wtp_dt.fit(wtp_train_SX, wtp_train_y)


# In[52]:


# predict and evaluate performance

wtp_dt_predictions = wtp_dt.predict(wtp_test_SX)

print(classification_report(wtp_test_y,wtp_dt_predictions, target_names=wtp_label_names))


# In[63]:


cohen_kappa_score(wtp_test_y,wtp_dt_predictions)


# In[45]:


#  feature importance scores based on the patterns learned by the model.

wtp_dt_feature_importances = wtp_dt.feature_importances_
wtp_dt_feature_names, wtp_dt_feature_scores = zip(*sorted(zip(wtp_feature_names,
wtp_dt_feature_importances), key=lambda x: x[1]))
y_position = list(range(len(wtp_dt_feature_names)))
plt.barh(y_position, wtp_dt_feature_scores, height=0.6, align='center')
plt.yticks(y_position , wtp_dt_feature_names)
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
t = plt.title('Feature Importances for Decision Tree')


# In[46]:


# ---


# #### 3-) Random Forest 

# In[57]:


from sklearn.ensemble import RandomForestClassifier


# In[58]:


wtp_rf = RandomForestClassifier(random_state=1)


# In[59]:


# Fitting the model

wtp_rf.fit(wtp_train_SX, wtp_train_y)


# In[60]:


# predict and evaluate performance

wtp_rf_predictions = wtp_rf.predict(wtp_test_SX)

print(classification_report(wtp_test_y,wtp_rf_predictions, target_names=wtp_label_names))


# In[64]:


cohen_kappa_score(wtp_test_y,wtp_rf_predictions)


# In[61]:


#  feature importance scores based on the patterns learned by the model.

wtp_rf_feature_importances = wtp_rf.feature_importances_
wtp_rf_feature_names, wtp_rf_feature_scores = zip(*sorted(zip(wtp_feature_names,
wtp_rf_feature_importances), key=lambda x: x[1]))
y_position = list(range(len(wtp_rf_feature_names)))
plt.barh(y_position, wtp_rf_feature_scores, height=0.6, align='center')
plt.yticks(y_position , wtp_rf_feature_names)
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
t = plt.title('Feature Importances for Decision Tree')


# In[62]:


# ---


# In[ ]:




