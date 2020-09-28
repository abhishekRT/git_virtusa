### import packages
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, precision_recall_curve, average_precision_score 
from sklearn.preprocessing import StandardScaler
def read_data(tp = "Train", N = 1542865627584):
    target = pd.read_csv(r'data\Train.csv'.format(tp.title(), N))
    pt = pd.read_csv(r'data\BeneficiaryData.csv'.format(tp.title(), N))
    in_pt = pd.read_csv(r'data\InpatientData.csv'.format(tp.title(), N))
    out_pt = pd.read_csv(r'data\OutpatientData.csv'.format(tp.title(), N))
    return (in_pt, out_pt, pt, target)
### Load Train data
train_inpt, train_outpt, train_asl, train_target = read_data()
t_male=train_asl[train_asl['Gender']==1]['Gender'].count()
t_female=train_asl[train_asl['Gender']==2]['Gender'].count()
t_total=train_asl['Gender'].count()
fraud_tot=train_target['PotentialFraud'].count()
train_fraudulent=train_target[train_target['PotentialFraud']=='Yes']['PotentialFraud'].count()
train_Nonfraudulent=fraud_tot-train_fraudulent
details=dict()
details['t_total-']=t_total
details['t_male-']=t_male
details['t_female-']=t_female
details['Provider_count-']=fraud_tot,
details['fraudulent-']=train_fraudulent
details['non fraudulent-']=train_Nonfraudulent
def read_data1(tp = "Test", N = 1542969243754):
    target1 = pd.read_csv(r'data\Test.csv'.format(tp.title(), N))
    pt1 = pd.read_csv(r'data\BeneficiaryDataTest.csv'.format(tp.title(), N))
    in_pt1 = pd.read_csv(r'data\InpatientDataTest.csv'.format(tp.title(), N))
    out_pt1 = pd.read_csv(r'data\OutpatientDataTest.csv'.format(tp.title(), N))
    return (in_pt1, out_pt1, pt1, target1)
### Load Test data
test_inpt, test_outpt, test_asl, test_target = read_data1()
def uniq(a):
    return np.array([len(set([i for i in x[~pd.isnull(x)]])) for x in a.values])
def Preprocessing(in_pt,out_pt,asl,target,mode):
    ### replacing 2 with 0 to represent the data in 1's and 0's
    asl = asl.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                           'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2, 
                           'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2, 
                           'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2, 'Gender': 2 }, 0)
    asl = asl.replace({'RenalDiseaseIndicator': 'Y'}, 1).astype({'RenalDiseaseIndicator': 'int64'})
    ### Determining whether patient dead or live
    asl['WhetherDead']= 0
    asl.loc[asl.DOD.notna(),'WhetherDead'] = 1
    if mode=='train':
        target["target"] = np.where(target.PotentialFraud == "Yes", 1, 0) 
    ### merging inpatient and outpatient data into Medicare
    MediCare = pd.merge(in_pt, out_pt, left_on = [ x for x in out_pt.columns if x in in_pt.columns], right_on = [ x for x in out_pt.columns if x in in_pt.columns], how = 'outer')
    ### merging medicare with benificiary data into data
    data = pd.merge(MediCare, asl,left_on='BeneID',right_on='BeneID',how='inner')
    ### Extra claims check
    ClmProcedure_vars = ['ClmProcedureCode_{}'.format(x) for x in range(1,7)]
    data['NumProc'] = data[ClmProcedure_vars].notnull().to_numpy().sum(axis = 1)
    keep = ['BeneID', 'ClaimID', 'ClmAdmitDiagnosisCode', 'NumProc' ] + ClmProcedure_vars
    data = data.drop(ClmProcedure_vars, axis = 1)
    ClmDiagnosisCode_vars =['ClmAdmitDiagnosisCode'] + ['ClmDiagnosisCode_{}'.format(x) for x in range(1, 11)]
    data['NumClaims'] = data[ClmDiagnosisCode_vars].notnull().to_numpy().sum(axis = 1)
    data['NumUniqueClaims'] = uniq(data[ClmDiagnosisCode_vars])
    data['ExtraClm'] = data['NumClaims'] - data['NumUniqueClaims']
    ### Hospitalization flag 
    data = data.drop(ClmDiagnosisCode_vars, axis = 1)
    data = data.drop(['NumClaims'], axis = 1)
    data['Hospt'] = np.where(data.DiagnosisGroupCode.notnull(), 0, 1)
    data = data.drop(['DiagnosisGroupCode'], axis = 1)
    ### converting all dates in data into spcific format for processing  
    data['AdmissionDt'] = pd.to_datetime(MediCare['AdmissionDt'] , format = '%Y-%m-%d')
    data['DischargeDt'] = pd.to_datetime(MediCare['DischargeDt'],format = '%Y-%m-%d')
    data['ClaimStartDt'] = pd.to_datetime(data['ClaimStartDt'] , format = '%Y-%m-%d')
    data['ClaimEndDt'] = pd.to_datetime(data['ClaimEndDt'],format = '%Y-%m-%d')
    data['DOB'] = pd.to_datetime(asl['DOB'] , format = '%Y-%m-%d')
    data['DOD'] = pd.to_datetime(asl['DOD'],format = '%Y-%m-%d')
    data['AdmissionDays'] = ((data['DischargeDt'] - data['AdmissionDt']).dt.days) + 1
    data['ClaimDays'] = ((data['ClaimEndDt'] - data['ClaimStartDt']).dt.days) + 1
    data['Age'] = round(((data['ClaimStartDt'] - data['DOB']).dt.days + 1)/365.25)
    ### false billing if any action taken place after patients death date.
    data['DeadActions'] = np.where(np.any(np.array([ data[x] > data['DOD'] for x in ['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt']]), axis = 0), 1, 0)
    data = data.fillna(0).copy()
    df1 = data.groupby(['Provider'], as_index = False)[['InscClaimAmtReimbursed', 'DeductibleAmtPaid', 'RenalDiseaseIndicator', 
                                                  'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
                                                  'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 
                                                  'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 
                                                  'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 
                                                  'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
                                                  'ChronicCond_stroke', 'WhetherDead', 
                                                  'NumProc','NumUniqueClaims', 'ExtraClm', 'AdmissionDays',
                                                  'ClaimDays', 'Hospt']].sum()
    df2 = data[['BeneID', 'ClaimID']].groupby(data['Provider']).nunique().reset_index()
    df3 = data.groupby(['Provider'], as_index = False)[['NoOfMonths_PartACov', 'NoOfMonths_PartBCov',
                                                    'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
                                                    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'Age']].mean()
    df = df2.merge(df1, on='Provider', how='left').merge(df3, on='Provider', how='left')
    return df,target,data
train_model,target,train_result=Preprocessing(train_inpt, train_outpt, train_asl, train_target,'train')
train_model=pd.DataFrame(train_model)
train_result=pd.DataFrame(train_result)
target=pd.DataFrame(target)
test_model,tar,test_result=Preprocessing(test_inpt, test_outpt, test_asl, test_target,'test')
test_model=pd.DataFrame(test_model)
test_result=pd.DataFrame(test_result)
target1=pd.DataFrame(tar)
### Only Train dataset is labeled that why we split it to two sets train and validation
X_train, X_val, y_train, y_val = train_test_split(train_model.drop(['Provider'], axis = 1), target.target.to_numpy(), test_size=0.25, random_state=1)
X1_train, X1_val, y1_train, y1_val = train_test_split(test_model.drop(['Provider'], axis = 1), target.target.to_numpy()[:1353], test_size=0.00000000000000001, random_state=1)
cols = X_train.columns
X_train = StandardScaler().fit_transform(X_train)
X_val = StandardScaler().fit_transform(X_val)
X1_train= StandardScaler().fit_transform(X1_train)
class MasterL:
    
    def __init__(self, model, #### model is a method which we are going to use for detecting FRAUDS. For example: sklearn.svm
                 X= X_train, y= y_train, test= X_val, ### data
                 **kvars  #### additional key parameters for model
                ):
        self.clf = model( **kvars)
        self.methodname = model.__name__
        self.X_train = X
        self.y_train = y
        self.X_test = test
        self.fit(self.X_train, self.y_train)
        self.predicted = self.predict(test)
        
    def fit (self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, x):
        return self.clf.predict(x)
       
    def get_score(self, y = y_val, roc = True, params = False):
        accuracy = accuracy_score(self.predicted, y)
        if params:
            print(self.clf.get_params())
        print(self.methodname+ " metrics:\n")
        print(" Accuracy Score: %.2f%%" % (accuracy * 100.0))
        print(" Confusion matrix:", "\n",confusion_matrix(y_true=y, y_pred=self.predicted))
        print( 'Classification report:\n', classification_report(y, self.predicted))
        if roc:
            print(" ROC Score: %.2f%%" % (roc_auc_score(y, self.clf.predict_proba(self.X_test)[:,1])))
### Logistic regression 
### Balanced Weight and Scaled data
ML1 = MasterL(LogisticRegression, 
              penalty= 'l1',
              solver= 'liblinear', class_weight='balanced', random_state = 5 , C = 0.001)
ML1.get_score()
### using logistic Regression detecting the fraud in test data
algo=LogisticRegression(penalty = 'l1', solver= 'liblinear', class_weight='balanced', random_state = 5)
algo.fit(X_train,y_train)
kp=algo.predict(X1_train)
kp=kp.astype('str')
kp=np.char.replace(kp,'0','Not Happened')
kp=np.char.replace(kp,'1','Happened')
target1=target1[:-1]
kp=pd.DataFrame(data=kp,columns=['Fraud'])
target1=pd.DataFrame(target1)
if 'Fraud Result' not in target1.columns:
    target1=pd.merge(target1,kp,how="outer",left_on=None, right_on=None, left_index=True, right_index=True)
def fraud(provider_no):
    frauds_list=[]
    provider_no=provider_no.strip()
    unbundle,false_bill=0,0
    fraud_count=dict()
    if(not(len(provider_no)==8 and provider_no[:3]=='PRV' )):
        return ["INVALID PROVIDER NUMBER"],fraud_count
    try:
        idx=(test_result.index[test_result['Provider']==provider_no])
    except:
        return ["INVALID PROVIDER NUMBER"],fraud_count
    for i in range(len(idx)):
        claim=[]
        claim.append(test_result['ClaimID'][idx[i]])
        if test_result['DeadActions'][idx[i]]==1:
            claim.append('False Billing')
            false_bill+=1
        if test_result['ExtraClm'][idx[i]]==1:
            claim.append('Unbundling')
            unbundle+=1
        if len(claim)==1:
            claim.append('Fruad Not Happened')
        frauds_list.append(claim)
        fraud_count['False Billing']=false_bill
        fraud_count['Unbundling']=unbundle
    return frauds_list,fraud_count
#lp,dp=fraud("PRV57070")
# for i in lp:
#     print(*i)
# for cou in dp.items():
#     print(cou)
def Claim_level(Claim_id):
    Cliam_id = Claim_id.strip()
    claim=[]
    try:
        idx=(test_result.index[test_result['ClaimID']==Claim_id])[0]
    except:
        return [0,0]
    claim.append(int(test_result['Age'][idx]))
    if(claim[0]==0):claim[0]='Age Not mentioned'
    if test_result['DeadActions'][idx]==1:
        claim.append('False Billing')
    if test_result['ExtraClm'][idx]==1:
        claim.append('Unbundling')
    if len(claim)==1:
        claim.append('Fruad Not Happened')
    return claim
# print(Claim_level('CLM63574'))
def chart():
    return details
show=chart()