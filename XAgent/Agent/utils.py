"""bla"""
# from __future__ import print_function
import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
# import string
import os
import sys

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./'):

    if dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 5, 6, 7, 8, 9, 13]

        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=None)
    return dataset



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = list(range(data.shape[1]))
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx
    ret.data = data
    return ret
