## High Performance Computing with Random Forest

### File Instructions

There are 3 files in the folder:

`data.csv` contains the synthetic data with 10 numeric features and 1 binary label, and there are around 50000 data.

`random_forest.py` is the implementation of **decision tree** and **random forest** without using any machine-learning libraries. 
  
  For **decision tree**, It follows the procedure of: 
    
  1) Calculate information gain. 
  2) Split the data via finding the best feature based on information gain. 
  3) Create branches recursivly based on the best feature until every leaf contains only a single category. 
    
  And for **random forest**, It follows the procedure of: 
  
  1) Bagging. Perform 100 bootstrapping on the data, generate different decision trees, and perform majority-voting on their prediction results. 
  2) When each node of the decision tree is split, 3 features are randomly selected in the way of non-replacement sampling, and split based on best feature is performed accordingly.

`MPI_random_forest.py` is the implementation of parallel computing based on MPI, accelerating the procedure of generating different subtrees when performing bootstrapping.

  The main parallel part in the code is:

  1) First, calculating how many bootstraps should be assigned to a process, and in each process, generated trees accordingly, and then gather all the trees generated into a list in rank 0.
  2) When calculating how many bootstraps should be assigned to a process, there may appear a situation that bootstrap cannot be assigned to every process without remainders. In this code, for example, there are 100 bootstrap and 7 processes, then I will assign each process with following bootstraps: rank0=14, rank1=14, rank2=14, rank3=14, rank4=14, rank5=15, rank6=15.

  The command for calling different number of processors is:

    mpirun -n <number of processors> python MPI_random_forest.py

### Result Analysis

I had checked the `random_forest.py` accuracy result with `from sklearn.ensemble import RandomForestClassifier`, which is roughly the same. Howeverm the `random_forest.py` is quite slow, requesting for around 20 mintues to finsih running.

For the multi-processor results to spped up the runtime, the result is shown below:

| Num. of processors | 1st run | 2nd run | 3rd run | 4th run | 5th run | average |
|:-------------------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 1 | 1149.6 | 1127.5 | 1153.8 | 1219.3 | 1197.4 | 1169.52 |
| 2 | 954.8 | 964.3 | 949.6 | 927.1 | 968.0 | 952.76 |
| 3 | 857.8 | 844.9 | 852.6 | 861.9 | 868.1 | 857.06 |
| 4 | 732.4 | 727.7 | 739.0 | 749.8 | 736.3 | 737.04 |
| 5 | 695.5 | 709.2 | 703.9 | 691.1 | 693.5 | 698.64 |
| 6 | 678.2 | 673.3 | 682.1 | 691.9 | 669.7 | 679.04 |
| 7 | 652.8 | 640.2 | 645.3 | 649.1 | 663.1 | 650.1 |
| 8 | 647.2 | 643.8 | 639.9 | 642.4 | 646.1 | 643.88 |

When using the formula of $speedup = \frac{T_1}{T_p}$ to calculate the speedup of accelerated algorithm, where $T_1$ is the execution time on a single processor and $T_p$ is the execution time on p processors in parallel. The form of estimated sppedup value is shown in the below table:

| Number of processors | Speed up ratio |
|:---------------------:|:----------------:|
| 2 | 1.2275 |
| 3 | 1.3646 |
| 4 | 1.5868 |
| 5 | 1.6740 |
| 6 | 1.7223 |
| 7 | 1.7990 |
| 8 | 1.8164 |

And the corresponding plot of speedup is shown below:

<div align='center'>
  <img width="492" alt="截屏2024-11-26 18 24 14" src="https://github.com/user-attachments/assets/d5fe0028-7071-4158-a9da-1dbde3892122">
</div>
