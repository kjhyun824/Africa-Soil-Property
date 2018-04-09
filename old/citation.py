import numpy as np 
import pandas as pd
import csv as csv 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn import linear_model
#from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline 
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork 
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, f1_score 

citation = pd.read_csv("data_processing.csv")
#citation = pd.read_csv("pythondata_SVC.csv")

citation.loc[citation["isConinvention"]=="T","isConinvention"]=1
citation.loc[citation["isConinvention"]=="F","isConinvention"]=0

#five un_processed data 
citation["nTCCount"] =citation["nTCCount"].fillna(citation["nTCCount"].median())
citation["nAssigneeCount"]= citation["nAssigneeCount"].fillna(citation["nAssigneeCount"].median())

citation["nTC_Main"]=citation["nTC_Main"].fillna(citation["nTC_Main"].median())
citation["Assignee_Main"]=citation["Assignee_Main"].fillna(citation["Assignee_Main"].median())
citation["Assignee_Mainline"]=citation["Assignee_Mainline"].fillna(citation["Assignee_Mainline"].median())
################
citation["nTCCount"] = citation["nTCCount"].astype(int)
citation["nAssigneeCount"] = citation["nAssigneeCount"].astype(int)
citation["nTC_Main"] = citation["nTC_Main"].astype(int)
citation["Assignee_Main"] = citation["Assignee_Main"].astype(int)
citation["Assignee_Mainline"] = citation["Assignee_Mainline"].astype(int)


citation.loc[citation["USPC_Main"]==0,"mainOriginality"] =4

#Main
#citation.loc[citation["USPC_Main"]==0,"nTC_Main"] =0
#citation.loc[citation["USPC_Main"]==0,"Assignee_Main"] =0

citation = citation.convert_objects(convert_numeric=True)

#for previous data
predictors = ["nAssigneeCount","nTC_Main","Assignee_Main","Assignee_Mainline","nTCCount","USPC_Main","BackCount","NPL","TCT","mainOriginality","MainlineOriginality","IndependantClaim","DependantClaim","realinventor","nIPC","nFamily","isConinvention"]
#predictors = ["nAssigneeCount","nTC_Main","Assignee_Main","Assignee_Mainline","nTCCount","USPC_Main","BackCount","NPL","TCT","mainOriginality","MainlineOriginality","IndependantClaim","DependantClaim","nInventor","nIPC","nFamily","isConinvention"]



alg1= RandomForestClassifier(random_state=7,n_estimators=500,criterion="entropy",max_depth=8) 
alg2= tree.DecisionTreeClassifier(random_state=7, criterion="entropy",max_depth=8 ) 
alg3= SVC(random_state=7,gamma=0.2)
alg5 = linear_model.LogisticRegression()

#scores = cross_validation.cross_val_score(alg1, citation[predictors],citation["gen3"],cv=10)

#print(scores.mean())

kf = KFold(citation.shape[0], n_folds =10, random_state=7)
predictions = [] 
for train, test in kf:
	train_predictors = (citation[predictors].iloc[train,:])
	train_target = citation["gen3"].iloc[train]
	alg1.fit(train_predictors, train_target) 
	test_predictions = alg1.predict(citation[predictors].iloc[test,:]) 
	predictions.append(test_predictions) 
	#print confusion_matrix(test_predictions, train_target) 
	#print f1_score(test_predictions, train_target) 

predictions = np.concatenate(predictions, axis=0) 

submission=pd.DataFrame(predictions)
submission.to_csv("predictionsGen3.csv") 
submission=pd.DataFrame(citation["gen3"])
submission.to_csv("originalGen3.csv") 

