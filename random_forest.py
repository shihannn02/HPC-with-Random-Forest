import pandas as pd
import numpy as np
from math import log
import random
import time

pd.options.mode.chained_assignment = None

# define the function of Shannon entropy
def entopy(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))

# define the function of information gain of each column
def information_gain(data, col):
    # calculate conditional information entropy
    group_col = data.groupby(col).apply(lambda x: entopy(x['label']))
    prob = pd.value_counts(data[col]) / len(data[col])
    entropy_tol = sum(group_col * prob)
    entropy_label = entopy(data['label'])
    # return information gain
    return entropy_label - entropy_tol

# choose the best information gain using the above function and return the best feature
def choose_best_feature(data):
    # initialize the maximum information gain
    max_info_gain = 0

    # traverse all the columns to find the maximum infomation gain
    for col in data.columns[:-1]:
        # calculate information gain of current feature
        info_gain = information_gain(data, col)
        # find the maximum information gain
        if info_gain > max_info_gain:
            # store the maximum information gain
            max_info_gain = info_gain
            # store the current feature which reaches the maximum information gain
            max_col = col
    # return the best feature
    return max_col

# split the data into 2 parts, and find the midpoint as the bin
def binning(train_data, best_feature_col):
    # find the minimal train data and maximal train data
    train_min = train_data[best_feature_col].min()
    train_max = train_data[best_feature_col].max()

    # choose the midpoint number as the split condition
    threshold = (train_max+train_min)/2

    # use the midpoint to split data as less than and larger to
    best_bins = np.array([train_min-100, threshold, train_max+100])

    return best_bins

# create a tree and record the data
def create_tree(dataSet, labels):
    # store labels
    train_label = dataSet[['label']]

    # If the labels in current data are of the same class, then return current class
    if len(np.unique(train_label)) == 1:
        return train_label.iloc[0, 0]

    # randomly select 3 features
    random_index = random.sample(range(len(labels) - 1), 3)
    random_label = [labels[i] for i in random_index]
    random_label.append('label')

    # choose the label with maximum information gain
    best_feature_label = choose_best_feature(dataSet[[i for i in random_label]])

    # In this feature, find the best binning interval, and split the data
    best_bin = binning(dataSet, best_feature_label)

    # resplit the data using the current best binning interval
    dataSet['best_bin'] = pd.cut(x=dataSet[best_feature_label], bins=best_bin)

    # a dictionary that used to store the decision tree
    my_tree = {best_feature_label: {}}

    # find subdata groups
    for attr, subgroup in dataSet.groupby(by='best_bin'):
        # store sublabels
        slabels = labels[:]

        # If there is no data in subgroup, then return the labels mostly appeared in last node
        if len(subgroup) == 0:
            my_tree[best_feature_label][attr] = train_label['label'].mode()[0]
        else:
            # throw the feature that has been used
            new_data = subgroup.drop(['best_bin'], axis=1)
            # Use recursive to build the tree for subdata
            my_tree[best_feature_label][attr] = create_tree(new_data, slabels)
    return my_tree

# randomly choose 1000 data to feed into the tree
def split_1000_data(train_data):
    length_1 = len(train_data)
    # set train data size as 1000
    train_len = 1000
    # record the index of train data, for further selecting from train data
    train_index = train_data.index

    # randomly generate the index of data
    random_index = random.sample(range(1, length_1), train_len)
    # select index randomed before from index of train data
    selected_index = [train_index[i] for i in random_index]

    # select the data for bagging tree
    boost_data = train_data.loc[train_data.index.isin(selected_index)]

    return boost_data

# generate 100 trees in random forest accordingly
def random_forest(train_data):
    # need to generate 100 bootstrap trees
    tree_ = 100
    # Now we'll process all trees in one thread
    tree_num = tree_

    # a empty list to store all the trees
    tree_res = []

    # start generating the tree
    for i in range(tree_num):
        labels = train_data.columns.to_list()

        # This is the data that will be feeded into the model in ith round
        ith_data = split_1000_data(train_data)

        # train the multilayer decision tree with random selected features
        my_tree = create_tree(ith_data,labels)

        # append the generated tree to the list
        tree_res.append(my_tree)

    return tree_res

# predict the label of the input test data
def classify(inputTree, labels, testVec):
    # store the first node in the generated tree
    firstStr = next(iter(inputTree))
    # store the next dictionary
    secondDict = inputTree[firstStr]
    # store the keys of the next dictionary
    key_list = list(secondDict.keys())

    for key in range(len(key_list)):
        # if the test data belongs to key range, then step into the next dictionary
        if testVec[firstStr] > key_list[key].left and testVec[firstStr] <= key_list[key].right:
            if type(secondDict[key_list[key]]) == dict:
                classLabel = classify(secondDict[key_list[key]], labels, testVec)
            else:
                # step into the next dictionary
                classLabel = secondDict[key_list[key]]
    return classLabel

# calculate the accuracy of the test data
def acc_classify_rf(tree_res,train_data,test_data):
    # all the feature names in the data
    labels = list(train_data.columns)
    result = []

    for i in range(len(test_data)):
        # this list is to record the precition result for each tree with the same data
        each_level = []
        test = test_data.iloc[i, :-1]

        # record the data into the list
        for j in range(len(tree_res)):
            each_level.append(classify(tree_res[j], labels, test))

        # choose the maximum vote label
        max_vote = max(each_level,key=each_level.count)
        # append the prediction result into the list
        result.append(max_vote)

    # add the prediction result into the last column of data
    test_data['predict']=result
    # calculate the accuracy
    acc = (test_data.iloc[:,-1]==test_data.iloc[:,-2]).mean()
    return acc

def spilt_data(selected_data):
    # calculate the length of the data, and choose 0.8 percent of data as train data
    length_1 = len(selected_data)
    train_len = int(length_1*0.8)

    # randomly select 800 index as train data
    train_index = random.sample(range(1,length_1),train_len)
    # this is train data and test data
    train_data = selected_data.loc[selected_data.index.isin(train_index)]
    test_data = selected_data.loc[~(selected_data.index.isin(train_index))]

    return train_data,test_data

# read data from the csv file
data = pd.read_csv('data.csv')
# split train data and test data
train_data,test_data = spilt_data(data)

# Start timing
start = time.time()

# train the model for random forest
model = random_forest(train_data)

# Evaluate the model
evaluate = acc_classify_rf(model, train_data, test_data)
end = time.time()
print(f'total running time is: {end-start}')
print(f'mean accuracy for random forest is: {evaluate}')