import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import copy
from joblib import dump
from joblib import load
import sys
import numpy as np
from sklearn import datasets

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label


def sample_digit_data(shape = 50):
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    x = data[np.random.choice(a=data.shape[0], size=shape, replace=False)]
    return x


def predict_class():
    model = load('svm_gamma=0.0001_C=4.joblib')
    x = sample_digit_data(shape=100)
    y_hyp = model.predict(x)
    unique_predicted_classes = np.unique(y_hyp)
    return unique_predicted_classes.shape[0]

def class_bias():
    unique_classes_count = predict_class()
    assert unique_classes_count != 1

TOTAL_CLASSES_TRAIN = 10
unique_classes_count = predict_class()
assert unique_classes_count == TOTAL_CLASSES_TRAIN