
# coding: utf-8

# In[1]:


#Importin Libraries

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc


# In[2]:


#Setting working Directory
os.chdir("C:\\Users\\DELL\\Desktop\\project")


# In[3]:


os.getcwd()


# In[4]:


train=pd.read_csv("Train_data.csv")
test=pd.read_csv("Test_data.csv")


# In[5]:


#Importing Data

train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")


# In[6]:


train.columns = train.columns.str.replace(' ','_')
test.columns = test.columns.str.replace(' ','_')


# In[7]:


original_train = train.copy()
original_test = test.copy()


# In[8]:


train.columns


# In[9]:


train.shape


# In[10]:


features = pd.DataFrame(train.columns)


# # Exploratory Data Analysis

# In[11]:


train.head()


# In[24]:


train.shape


# In[12]:


#Seperating Continous and categorical variables for analysis

cnames = ['account_length', 'number_vmail_messages', 'total_day_minutes', 'total_day_calls',
          'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
          'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes',
          'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']
cat_names = ['state','area_code','international_plan', 'voice_mail_plan']


# In[13]:


#Density plots of continous variables

fig,axes = plt.subplots(nrows = 5, ncols = 3, figsize = (32,36)) 
k=0
for i in range(5):
    for j in range(3):
        axes[i,j].hist(train[cnames[i+j+k]], bins =30)
        axes[i,j].set_title(cnames[i+j+k].replace('_',' '), fontsize = 25)
        axes[i,j].set_ylabel('Count', fontsize = 20)
    k=k+2
plt.tight_layout
#plt.savefig('Distributionplots.png')


# In[15]:


#Plotting 'number_vmail_messages' without zero value

df = train.loc[train['number_vmail_messages']>0,'number_vmail_messages']
plt.hist(df, bins = 20)
plt.ylabel('Count', fontsize = 20)
plt.xlabel('Messages', fontsize = 20)
#plt.savefig('voicemail.png')


# In[28]:


# Log transforming the skewed variables (if needed)

#for  i in ['number_vmail_messages', 'total_intl_calls', 'number_customer_service_calls']:
#    X = train[i].values + 1 
#    train[i] = np.log(X)


# In[29]:


# Z-Score transform (if needed)

#from scipy import stats
#train['number_vmail_messages'] = stats.zscore(train['number_vmail_messages'])


# In[30]:


#Checking corelations of continous variables

c_corr = train[cnames].corr()
plt.figure(figsize = (60,60))
sns.set(font_scale = 3.8)

sns.heatmap(c_corr, cmap='magma', linecolor='white', linewidth=5, square = True,
            xticklabels = list(pd.Index(cnames).str.replace('_',' ')),
            yticklabels = list(pd.Index(cnames).str.replace('_',' ')))
#plt.savefig('Corelations.png')


# In[31]:


#Checking dependency of dependent variable on categorical variables

for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train['Churn'], train[i]))
    print(p)


# In[32]:


#Checking counts of Target variable
plt.figure(figsize = (9,6))
sns.set(font_scale = 1)
sns.countplot(x = 'Churn', data = train)
plt.xlabel('Churn', fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
#plt.savefig('TargetCount')


# In[16]:


# Plot of Number of voicemail messages by Class

plt.figure(figsize = (10,15))
train.hist('number_vmail_messages', by = 'Churn')
plt.ylabel('Count', fontsize = 20)
#plt.savefig('voicemailClass.png')


# In[34]:


# Plot of Total Intl calls by Class

plt.figure(figsize = (10,15))
train.hist('total_intl_calls', by = 'Churn')
plt.ylabel('Count', fontsize = 20)
#plt.savefig('intlcallsClass')


# In[35]:


# Plot of Number of customer service calls by Class

plt.figure(figsize = (10,15))
train.hist('number_customer_service_calls', by = 'Churn')
plt.ylabel('Count', fontsize = 20)
#plt.savefig('servivecallsClass.png')


# In[36]:



# Plot of States

plt.figure(figsize = (15,10))
sns.countplot('state', data= original_train)
plt.xlabel('State', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
#plt.savefig('state.png')


# # Feature Scaling and Feature Selection

# In[17]:


# Dropping the irrelevant variables

drop_col = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes',
            'total_intl_minutes', 'area_code', 'phone_number']
train.drop(drop_col, axis = 1, inplace = True)
test.drop(drop_col, axis = 1, inplace = True)


# In[18]:


# Replacing 'Yes','No','True','False' with 1 and 0

train['international_plan'] = train['international_plan'].replace(' yes', 1).replace(' no', 0)
train['voice_mail_plan'] = train['voice_mail_plan'].replace(' yes', 1).replace(' no', 0)
train['Churn'] = train['Churn'].replace(' False.', 0).replace(' True.', 1)
test['international_plan'] = test['international_plan'].replace(' yes', 1).replace(' no', 0)
test['voice_mail_plan'] = test['voice_mail_plan'].replace(' yes', 1).replace(' no', 0)
test['Churn'] = test['Churn'].replace(' False.', 0).replace(' True.', 1)


# In[19]:


# Assigning a code to each state

keys = train['state'].unique().tolist()
values = list(range(len(keys)))
state_codes = dict(zip(keys,values))
train['state'] = train['state'].map(state_codes)
test['state'] = test['state'].map(state_codes)


# # Preparing Data for Models

# In[20]:


# Preparing Data for model training

train_var = train.columns
train_data_X = train[train_var].drop('Churn', axis = 1)
train_data_Y = train['Churn']
test_data_X = test[train_var].drop('Churn', axis = 1)
test_data_Y = test['Churn']


# In[21]:


get_ipython().system('pip install imblearn')


# In[22]:


#Over sampling the complete data to deal with target class imbalance

from imblearn.over_sampling import SMOTE
smote = SMOTE()

print("Before OverSampling, counts of label '1': {}".format(sum(train_data_Y==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(train_data_Y==0)))

train_data_X_over, train_data_Y_over = smote.fit_sample(train_data_X, train_data_Y.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(train_data_Y_over==1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(train_data_Y_over==0)))


# # Model Development

# In[58]:


# Custom Function for Accuracy and FNR

def conf_matrix(y,pred):
    CM = pd.crosstab(y,pred)
    
    Accuracy = (sum(np.diag(CM)) * 100)/len(pred)
    FNR = (CM.iloc[1,0] *100)/sum(CM.iloc[1,])
    
    #print(CM)
    #print('Accuracy : {:.3f}'.format(Accuracy))
    #print('FNR : {:.3f}'.format(FNR))
    return (Accuracy,FNR)


# In[59]:


# Custom function for auc

def auc_val(y,pred):
    fpr,tpr,thresholds = roc_curve(y,pred)
    roc_auc = auc(fpr,tpr)
    auc_4f = round(roc_auc,4)
    return (auc_4f)


# In[60]:


# Splitting the training data for model evaluation

X_train_under, X_valid, y_train_under, y_valid = train_test_split(train_data_X, train_data_Y,
                                                    stratify = train_data_Y, test_size = 0.2)


# In[61]:


#Over sampling for models to deal with target class imbalance

from imblearn.over_sampling import SMOTE
smote = SMOTE()

print("Before OverSampling, counts of label '1': {}".format(sum(y_train_under==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_under==0)))

X_train, y_train = smote.fit_sample(X_train_under, y_train_under.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))


# # Decision Tree

# In[62]:



#Decission Tree

from sklearn.tree import DecisionTreeClassifier


# In[63]:


# Grid Search to find best max_depth

best_fnr = 100
dt_train_auc = []
dt_test_auc = []

max_dep = [6, 8, 10, 12, 15, 18, 20]

for max_depth in max_dep:
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth,
                                 random_state = 0)
    clf.fit(X_train, y_train)
    temp1_pred = clf.predict(X_train)
    dt_train_auc.append(auc_val(y_train,temp1_pred))
    tr_acc = clf.score(X_train,y_train)
    print('Training Accuracy : {:.3f}'.format(tr_acc))
    temp2_pred = clf.predict(X_valid)
    dt_test_auc.append(auc_val(y_valid, temp2_pred))
    Acc,fnr = conf_matrix(y_valid, temp2_pred)
    print('---')
    if (((Acc > 80) & (tr_acc < 1)) & (fnr < best_fnr)):
        best_fnr = fnr
        best_params = {'max_depth': max_depth}

print('Best FNR : {:.2f}'.format(best_fnr))
print('Best_FNR_parameters : {}'.format(best_params))


# In[64]:


# AUC grids

dt_train_auc = pd.DataFrame(dt_train_auc, index = max_dep)
dt_test_auc = pd.DataFrame(dt_test_auc, index = max_dep)
print('Training AUC')
print(dt_train_auc)
print('Test AUC')
print(dt_test_auc)
dt_test_auc.to_csv('DT_AUC.csv')


# In[65]:


# Fitting the model with best parameters based on test_auc on complete training data

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, random_state = 0)

tree.fit(train_data_X_over, train_data_Y_over)


# In[66]:


# Training Accuracy

tree.score(train_data_X_over, train_data_Y_over)


# In[67]:


# Performance on Actual Test data

pred_dt = tree.predict(test_data_X)

Accuracy_dt,FNR_dt = conf_matrix(test_data_Y,pred_dt)

dt_auc = auc_val(test_data_Y,pred_dt)

print('Test Accuracy: {:.3f}'.format(Accuracy_dt))
print('Test FNR: {:.3f}'.format(FNR_dt))
print('Test AUC: {:.3f}'.format(dt_auc))


# # Random Forest

# In[68]:


#Random Forest

from sklearn.ensemble import RandomForestClassifier


# In[69]:


# Grid Search for finding best parameters

best_fnr = 100

n_estimators = [40, 60, 80, 100, 200]
m_depth = [6, 8, 10, 12, 15, 18, 20]

rf_train_auc = np.zeros((len(n_estimators),len(m_depth)))
rf_test_auc = np.zeros((len(n_estimators),len(m_depth)))

i = 0

for n_est in n_estimators:
    j = 0
    for max_d in m_depth:
        clf = RandomForestClassifier(n_estimators = n_est, max_features = 'sqrt',
                                     oob_score = True, max_depth = max_d, criterion = 'entropy',
                                     random_state = 0)
        clf.fit(X_train, y_train)
        temp1_pred = clf.predict(X_train)
        rf_train_auc[i,j] = auc_val(y_train, temp1_pred)
        tr_acc = clf.score(X_train,y_train)
        print('Training Accuracy : {:.3f}'.format(tr_acc))
        temp2_pred = clf.predict(X_valid)
        rf_test_auc[i,j] = auc_val(y_valid, temp2_pred)
        Acc,fnr = conf_matrix(y_valid, temp2_pred)
        print('---')
        if (((Acc > 80) & (tr_acc < 1)) & (fnr < best_fnr)):
            best_fnr = fnr
            best_params = {'max_depth' : max_d, 'n_estimators' : n_est}
        j = j+1
    i = i+1

print('Best FNR : {:.2f}'.format(best_fnr))
print('Best_FNR_parameters : {}'.format(best_params))


# In[46]:


# AUC grids

rf_train_auc = pd.DataFrame(rf_train_auc, index = n_estimators, columns = m_depth)
rf_test_auc = pd.DataFrame(rf_test_auc, index = n_estimators, columns = m_depth)
print('Training AUC')
print(rf_train_auc)
print('Test AUC')
print(rf_test_auc)
rf_test_auc.to_csv('RF_AUC.csv')


# In[70]:


# Retraining the model for full training data with best parameters based on test_auc

rf_tree = RandomForestClassifier(n_estimators = 80, max_features = 'sqrt', oob_score = True, max_depth = 6, criterion = 'entropy', random_state = 0)

rf_tree.fit(train_data_X_over, train_data_Y_over)


# In[71]:


# Training Score

rf_tree.score(train_data_X_over, train_data_Y_over)


# In[73]:


# Performance on test data

pred_rf = rf_tree.predict(test_data_X)

Accuracy_rf,FNR_rf = conf_matrix(test_data_Y,pred_rf)

rf_auc = auc_val(test_data_Y,pred_rf)

print('Test Accuracy: {:.3f}'.format(Accuracy_rf))
print('Test FNR: {:.3f}'.format(FNR_rf))
print('Test AUC: {:.3f}'.format(rf_auc))


# # Logistic regression

# In[74]:


#Logistic Regression

import statsmodels.api as sm

logit = sm.Logit(y_train, X_train).fit()

logit.summary()


# In[75]:


# Building ROC curve to decide the threshold value for classification

#from sklearn.metrics import roc_curve, auc
fpr,tpr,thresholds = roc_curve(y_valid, logit.predict(X_valid))
plt.figure(figsize = (9,6))
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.savefig('ROC.png')


# In[76]:


# Model evaluation

logit_y_test = pd.DataFrame()
logit_y_test['prob'] = logit.predict(X_valid)

logit_y_test['pred'] = 1
logit_y_test.loc[logit_y_test.prob < 0.4, 'pred'] = 0

Accuracy,FNR = conf_matrix(y_valid,logit_y_test['pred'])
auc_train = auc_val(y_valid,logit_y_test['pred'])
print('Validation Test Accuracy: {:.3f}'.format(Accuracy))
print('Validation Test FNR: {:.3f}'.format(FNR))
print('Validation Test AUC: {:.3f}'.format(auc_train))


# In[77]:


# Retraining the model for full training data

logit = sm.Logit(train_data_Y_over, train_data_X_over).fit()

logit.summary()


# In[78]:


# Performance on test data

pred = pd.DataFrame()
pred['prob'] = logit.predict(test_data_X)

pred['pred'] = 1
pred.loc[pred['prob'] < 0.4, 'pred'] = 0 

Accuracy_lr,FNR_lr = conf_matrix(test_data_Y,pred['pred'])

lr_auc = auc_val(test_data_Y, pred['pred'])

print('Test Accuracy: {:.3f}'.format(Accuracy_lr))
print('Test FNR: {:.3f}'.format(FNR_lr))
print('Test AUC: {:.3f}'.format(lr_auc))


# # Support vector Machine

# In[79]:


# Support vector Classifier

from sklearn.svm import SVC


# In[80]:


# Grid Search for best parameters

best_fnr = 100

c_val = [0.01, 0.1, 1, 10, 100]
g_val = [0.001, 0.01, 0.1, 1, 10]

svc_train_auc = np.zeros((len(c_val), len(g_val)))
svc_test_auc = np.zeros((len(c_val), len(g_val)))

i = 0

for c in c_val:
    j = 0
    for gamma in g_val:
        clf = SVC(kernel = 'rbf', C = c, gamma = gamma, random_state = 0)
        clf.fit(X_train, y_train)
        temp1_pred = clf.predict(X_train)
        svc_train_auc[i,j] = auc_val(y_train, temp1_pred)
        tr_acc = clf.score(X_train,y_train)
        print('Training Accuracy : {:.3f}'.format(tr_acc))
        temp2_pred = clf.predict(X_valid)
        svc_test_auc[i,j] = auc_val(y_valid, temp2_pred)
        Acc,fnr = conf_matrix(y_valid,temp2_pred)
        print('---')
        if (((Acc > 80) & (tr_acc < 1)) & (fnr < best_fnr)):
            best_fnr = fnr
            best_params = {'C' : c, 'gamma' : gamma}
        j = j+1
    i = i+1

print('Best FNR : {:.2f}'.format(best_fnr))
print('Best_parameters : {}'.format(best_params))


# In[81]:


# AUC grids

svc_train_auc = pd.DataFrame(svc_train_auc, index = c_val, columns = g_val)
svc_test_auc = pd.DataFrame(svc_test_auc, index = c_val, columns = g_val)
print('Training AUC')
print(svc_train_auc)
print('Test AUC')
print(svc_test_auc)
svc_test_auc.to_csv('SVC_AUC.csv')


# In[82]:


# Fitting over full training data wit best parameters based on test_auc

svc = SVC(kernel = 'rbf', C = 10, gamma = 0.001, random_state = 0)

svc.fit(train_data_X_over,train_data_Y_over)


# In[83]:


# Training Score

svc.score(train_data_X_over,train_data_Y_over)


# In[84]:


# Performance on test data

pred_svc = svc.predict(test_data_X)

Accuracy_svc,FNR_svc = conf_matrix(test_data_Y,pred_svc)

svc_auc = auc_val(test_data_Y,pred_svc)

print('Test Accuracy: {:.3f}'.format(Accuracy_svc))
print('Test FNR: {:.3f}'.format(FNR_svc))
print('Test AUC: {:.3f}'.format(svc_auc))


# # Gradient Boosted Classifier

# In[85]:


#Gradient Boosted Classifier

from sklearn.ensemble import GradientBoostingClassifier


# In[86]:


# Grid Search for finding best parameters

best_fnr = 100
l_rate = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
m_depth = [2, 4, 6, 8, 10, 12, 15]

gbc_train_auc = np.zeros((len(l_rate),len(m_depth)))
gbc_test_auc = np.zeros((len(l_rate),len(m_depth)))

i=0

for learn_rate in l_rate:
    j = 0
    for max_d in m_depth:
        clf = GradientBoostingClassifier(max_depth = max_d, learning_rate = learn_rate, random_state = 0, max_features = 'sqrt')
        clf.fit(X_train, y_train)
        temp1_pred = clf.predict(X_train)
        gbc_train_auc[i,j] = auc_val(y_train, temp1_pred)
        tr_acc = clf.score(X_train,y_train)
        print('Training Accuracy : {:.3f}'.format(clf.score(X_train,y_train)))
        temp2_pred = clf.predict(X_valid)
        gbc_test_auc[i,j] = auc_val(y_valid, temp2_pred)
        Acc,fnr = conf_matrix(y_valid, temp2_pred)
        print('---')
        if (((Acc > 80) & (tr_acc < 1)) & (fnr < best_fnr)):
            best_fnr = fnr
            best_params = {'max_depth' : max_d, 'learning_rate' : learn_rate}
        j = j+1
    i = i+1

print('Best FNR : {:.2f}'.format(best_fnr))
print('Best_FNR_parameters : {}'.format(best_params))


# In[87]:


# AUC grids

gbc_train_auc = pd.DataFrame(gbc_train_auc, index = l_rate, columns = m_depth)
gbc_test_auc = pd.DataFrame(gbc_test_auc, index = l_rate, columns = m_depth)
print('Training AUC')
print(gbc_train_auc)
print('Test AUC')
print(gbc_test_auc)
gbc_test_auc.to_csv('GBC_AUC.csv')


# In[88]:


# Fitting over complete training data with best parameters based on test_auc

gbc = GradientBoostingClassifier(max_depth = 6, learning_rate = 0.01, random_state = 0, max_features = 'sqrt')

gbc.fit(train_data_X_over,train_data_Y_over)


# In[89]:


# Training Score

gbc.score(train_data_X_over,train_data_Y_over)


# In[90]:


# Performance on test data

pred_gbc = gbc.predict(test_data_X)

Accuracy_gbc,FNR_gbc = conf_matrix(test_data_Y,pred_gbc)

gbc_auc = auc_val(test_data_Y,pred_gbc)

print('Test Accuracy: {:.3f}'.format(Accuracy_gbc))
print('Test FNR: {:.3f}'.format(FNR_gbc))
print('Test AUC: {:.3f}'.format(gbc_auc))


# # Final Result table

# In[91]:


result = pd.DataFrame()

result['Model'] = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'Support vector Classifier', 'Gradient Boosted Classifier']
result['Test Accuracy'] = [Accuracy_dt, Accuracy_rf, Accuracy_lr, Accuracy_svc, Accuracy_gbc]
result['False negative rate'] = [FNR_dt, FNR_rf, FNR_lr, FNR_svc, FNR_gbc]
result['Test AUC'] = [dt_auc, rf_auc, lr_auc, svc_auc, gbc_auc]
result
result.to_csv('Result.csv')


# # Output using Selected Model i.e. Random Forest

# In[92]:


pd.DataFrame(pred_rf).to_csv('Test_data_Predictions.csv')

