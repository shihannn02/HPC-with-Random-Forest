## High Performance Computing with Random Forest

### File Instructions

There are 3 files in the folder:

`data.csv` contains the synthetic data with 10 numeric features and 1 binary label.

`random_forest.py` is the implementation of **decision tree** and **random forest** without using any machine-learning libraries. For **decision tree**, It follows the procedure of: 1) Calculate information gain. 2) Split the data via finding the best feature based on information gain. 3) Create branches recursivly based on the best feature until every leaf contains only a single category. And for **random forest**, It follows the procedure of: 1) Bagging. Perform 100 bootstrapping on the data, generate different decision trees, and perform majority-voting on their prediction results. 2) When each node of the decision tree is split, 3 features are randomly selected in the way of non-replacement sampling, and split based on best feature is performed accordingly.
