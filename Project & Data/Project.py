import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

warnings.filterwarnings('ignore')

print('---review Dataset---')

Data=pd.read_csv('bank-full.csv',sep=';')

print('---Data Information---')
print(Data.head())
print(Data.info())
print(Data.shape)
print(list(Data.columns))
print(Data.isnull().sum())

print('---Data Statistic Summary---')
print(Data.describe())


print('---Encoding Some Labels---')

le=LabelEncoder()
Data['default']=le.fit_transform(Data['default'])
Data['loan']=le.fit_transform(Data['loan'])
Data['housing']=le.fit_transform(Data['housing'])
Data['y']=le.fit_transform(Data['y'])

encoder=OneHotEncoder(sparse_output=False)
encoded=encoder.fit_transform(Data[['job', 'marital', 'education',
 'contact', 'month', 'poutcome']])
encoded_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(['job', 'marital', 
'education', 'contact', 'month', 'poutcome']))

numerical=Data[['age', 'balance', 'day', 'duration',
     'campaign', 'pdays', 'previous'
]]


print('---Visualization---')

plt.figure(figsize=(10,10))
sns.heatmap(Data.corr(numeric_only=True),cmap='coolwarm',annot=True)
plt.savefig('Corr Plot.png')
plt.show()

plt.figure(figsize=(10,10))
sns.barplot(x=Data['y'],y=Data['pdays'])
plt.title('Pdays---Y')
plt.savefig('Pdays---Y')
plt.show()


plt.figure(figsize=(10,10))
sns.barplot(x=Data['y'],y=Data['duration'])
plt.title('Duration---Y')
plt.savefig('Duration---Y')
plt.show()



plt.figure(figsize=(10,10))
sns.barplot(x=Data['y'],y=Data['balance'])
plt.title('Balance---Y')
plt.savefig('Balance---Y')
plt.show()


sns.pairplot(Data[['balance','duration','pdays','y']])
plt.savefig('pairplot.png')
plt.show()

Data=pd.get_dummies(Data,drop_first=True)
print(Data)


x=Data.drop(columns=['y'])
y=Data['y']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print(Data['y'].value_counts(normalize=True)*100)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

smote=SMOTE(random_state=42)
x_train_smote,y_train_smote=smote.fit_resample(x_train_scaled,y_train)


models={
    'Logestic':LogisticRegression(max_iter=1000),
    'KNN':KNeighborsClassifier(n_neighbors=5),
    'DecisionTree':DecisionTreeClassifier(random_state=42),
    'RandomForest':RandomForestClassifier(random_state=42),
    'GradientBoosting':GradientBoostingClassifier(random_state=42),
    'AdaBoost':AdaBoostClassifier(random_state=42),
    'XGboost':XGBClassifier()

}

results=[]

for name , model in models.items():
    if name in ['Logestic','KNN']:
        model.fit(x_train_scaled,y_train)
        prediction=model.predict(x_test_scaled)
    else:
        model.fit(x_train_smote,y_train_smote)
        prediction=model.predict(x_test_scaled)
    results.append({
        'Name':name,
        'Model':model,
        'ACC':accuracy_score(y_test,prediction),
        'RECALL':recall_score(y_test,prediction),
        'PREC':precision_score(y_test,prediction),
        'F1':f1_score(y_test,prediction)
    })

results=pd.DataFrame(results)
print(results.sort_values('F1',ascending=False))

print('CM:',confusion_matrix(y_test,prediction))
print('Report:',classification_report(y_test,prediction))



param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3],
    'subsample': [0.8, 1.0]
}


grid_gb=GridSearchCV(GradientBoostingClassifier(),cv=5,scoring='f1',
param_grid=param_grid_gb,verbose=2)

grid_gb.fit(x_train_smote,y_train_smote)
best_model=grid_gb.best_estimator_
best_predict=best_model.predict(x_test_scaled)
best_prob=best_model.predict_proba(x_test_scaled)[:, 1]
print('best parameters:',grid_gb.best_params_)
print('best F1 score:',f1_score(y_test,best_predict))
print('final Report:',classification_report(y_test,best_predict))


fpr,tpr,thersholds=roc_curve(y_test,best_prob)
AUC=roc_auc_score(y_test,best_prob)
print('AUC:',AUC)
plt.figure(figsize=(10,10))
plt.plot(fpr,tpr)
plt.title('AUC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.savefig('AUC curve.png')
plt.show()


importance=best_model.feature_importances_
plt.figure(figsize=(10,10))
plt.barh(range(len(importance)),importance)
plt.savefig('feature_importance.png')
plt.show()


pickle.dump(x.columns,open('Columns.pkl','wb'))
pickle.dump(best_model,open('Model.pkl','wb'))
pickle.dump(scaler,open('Scaler.pkl','wb'))
pickle.dump(smote,open('SMOTE.pkl','wb'))

print('Files Saved')



