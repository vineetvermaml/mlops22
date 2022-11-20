from sklearn import datasets, svm, metrics, tree
import pdb
import pandas as pd
import numpy as np 

from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    f1_scoring
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0004, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [5, 10, 25, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, f1_scoring]
h_metric = metrics.accuracy_score
predicted_lable_svm  = list()
predicted_labels_DecisionTree = list()
n_cv = 5
results = {}
for n in range(n_cv):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )

    models = {
        "svm": svm.SVC(),
        "decision_tree": tree.DecisionTreeClassifier(),
    }
    for clf_name in models:
        clf = models[clf_name]
        print("[{}] Running hyper parameter tuning for {} ====>> ".format(n,clf_name))
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
        )
        best_model = load(actual_model_path)

        predicted = best_model.predict(x_test)
        if clf_name =='svm':
    	       predicted_lable_svm = predicted
        if clf_name =='decision_tree':
    	       predicted_labels_DecisionTree = predicted
 
        if not clf_name in results:
            results[clf_name]=[]    

        results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})

        print(
            f"Classifier's Matrix ==== {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

#print(results)
df = pd.DataFrame(results)
print(df)

mean_svm=[]
for i in range(5):
    mean_svm.append(results['svm'][i]['accuracy_score'])
df1=pd.DataFrame(mean_svm)
print("SVM Accuracy Mean")
print(float(np.round(df1.mean(),4)))
print("SVM Standard Deviation")
print(float(np.round(df1.std(),4)))

mean_dt=[]
for i in range(5):
    mean_dt.append(results['decision_tree'][i]['accuracy_score'])
df2=pd.DataFrame(mean_dt)
print("DECISION TREE Accuracy Mean")
print(float(np.round(df2.mean(),4)))
print("DECISION TREE Standard Deviation")
print(float(np.round(df2.std(),4)))


count=0
for i in range(len(predicted_lable_svm)):
 	if predicted_lable_svm[i] != predicted_labels_DecisionTree [i]:
         count=count+1
print("Gap count in prediction label betweeen SVM and DECISION TREE",':',count)
