# Master's thesis about CL for IDS

Tested on **Python 3.12.0**.

Repository containing pipelines for preparing data and experimenting for my Master's thesis about Continual Learning in the Intrusion Detection Systems domain.

Cristian Piacente 866020 @ University of Milan - Bicocca

## How to Run

### Setup Virtual Environment

First of all, it's necessary to create a Python virtual environment and to install all the dependencies.

#### Creating and activating

Using the repository **root directory** as the working directory, execute the following command to **create the virtual environment**:

    python3 -m venv venv

Once the virtual environment has been created, it will be saved in the filesystem, so it won't be necessary to recreate it every time.

Now, it's possible to **activate it** with

    source ./venv/bin/activate

Please note: if you are on Windows, remove `source` from the command and use `Scripts` instead of `bin`.

#### Installing the dependencies

After the **first activation**, it's necessary to **install all the needed libraries** by executing

    pip3 install -r requirements.txt

Once the dependencies have been installed, it won't be necessary anymore for the future activations.
With the virtual environment activated, it's possible to execute the different parts of the project, described in the other sections.

#### Deactivating

To quit the virtual environment, it's sufficient to execute the command

    deactivate

### Pipeline

#### Design

##### DATA PREPARATION PIPELINE

**Inputs**: 

- dataset name (used for identifying the datasets folder, make sure you downloaded it from the [GDrive](https://drive.google.com/drive/folders/1auey1u1GrCB29wmuOXzskXgjSdlHo5f8?usp=sharing))
- file extension to consider (.csv/.parquet, used to identify the correct files in the folder)
- int threshold parameter (see 2a. in "Steps", default=3000)
- parameters for preprocessing:
    - list parameter of features to drop (default=[])
    - list parameter of attack types to drop (default=[])
- the sampling strategy used in the training set among all tasks ("natural", "balanced_attacks" or "fully_balanced")

**Steps**:

0. Merge all data (train-val-test) into one file if not already done
1. Preprocessing, but no normalization
2. Split into two distributions: 
    2a. Top classes (including benign) where number of samples >= threshold parameter (e.g. 3000), i.e. **Closed-Set** multi-classification (known classes)
    2b. The rest where we set the labels to "unknown", i.e. **Out-of-Distribution** binary classification (0-day attacks) 
3. Split 2a. (Closed-Set) into training (60%), validation (20%) and test set (20%)
4. Create tasks (batches) using the training set, each one with benign flows and 1 attack type flows, where sampling could be applied depending on the input parameter: if **"natural"**, no balancing is performed and for each task we use all the available attack samples w.r.t that class; if **"balanced_attacks"** every task will have the same number of attack samples using the minority class; if **"fully_balanced"** benign flows are balanced per task too, using the same number

###### Preprocessing steps

- automatic creation of columns attack_type, attack
- drop features in a list parameter
- drop constant features
- drop attack types in a list parameter
- data cleaning (drop duplicates, fix missing values, inf values, negative values)
- casting int types properly
- encoding for categorical features
- casting bool types properly 

##### EXPERIMENTING PIPELINE

**Inputs**: 

- the data preparation pipeline must have been executed
- the dataset name (used for identifying the datasets folder + creating one in the models folder)
- hyperparameters for tuning (see examples in the sections below)

**Steps (TODO)**:

6. 
    a. Baseline modeling and evaluation on the full Closed-Set (i.e., use the split dataset from 3.) \
    b. Modeling on each task from train (simulate a real-time data stream), without catastrophic forgetting: \
        - Incrementally train (without storing the previous data) \
        - Evaluate on the validation set and predict on the test set (Continual Learning while checking the ability of retaining past knowledge)
7. (TODO) Use the OoD data for testing generalization capabilities


#### Execution

##### DATA PREPARATION PIPELINE

With the virtual environment activated in the repository root, to execute the Data Preparation pipeline here are some examples.

The parameters are `--dataset-name`, `--input-file-extension`, `--features-to-drop`, `--attack-types-to-drop`, `--threshold`, `--sampling-strategy`.

The first command is common for every dataset: change the working directory to `data_prep_pipeline` with

    cd data_prep_pipeline

###### For CIC-IDS2017

    python3 -m luigi --module pipeline CreateTasks --dataset-name "CIC-IDS2017" --input-file-extension ".parquet" --features-to-drop "[\"flow_id\", \"src_addr\", \"src_port\", \"dst_addr\", \"dst_port\", \"timestamp\"]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler

###### For CSE-CIC-IDS2018

    python3 -m luigi --module pipeline CreateTasks --dataset-name "CSE-CIC-IDS2018" --input-file-extension ".csv" --features-to-drop "[\"Dst Port\", \"Timestamp\"]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler

###### For LYCOS-IDS2017

    python3 -m luigi --module pipeline CreateTasks --dataset-name "LYCOS-IDS2017" --input-file-extension ".parquet" --features-to-drop "[\"flow_id\", \"src_addr\", \"src_port\", \"dst_addr\", \"dst_port\", \"timestamp\"]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler

###### For NSL-KDD

    python3 -m luigi --module pipeline CreateTasks --dataset-name "NSL-KDD" --input-file-extension ".csv" --features-to-drop "[]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler

###### For UNSW-NB15

    python3 -m luigi --module pipeline CreateTasks --dataset-name "UNSW-NB15" --input-file-extension ".csv" --features-to-drop "[\"id\"]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler

###### For TON-IoT

    python3 -m luigi --module pipeline CreateTasks --dataset-name "TON-IoT" --input-file-extension ".csv" --features-to-drop "[\"ts\", \"src_ip\", \"src_port\", \"dst_ip\", \"dst_port\", \"dns_query\", \"uid\", \"ssl_subject\", \"ssl_issuer\", \"http_method\", \"http_uri\", \"http_version\", \"http_user_agent\", \"http_orig_mime_types\", \"weird_notice\"]" --attack-types-to-drop "[]" --threshold 3000 --sampling-strategy "natural" --local-scheduler


##### EXPERIMENTS PIPELINE

With the virtual environment activated in the repository root, to execute the Experiments pipeline here are some examples.

Please note that this pipeline must be executed **after** the Data Preparation pipeline, in order to have the input files.

The examples are shown for the dataset CIC-IDS2017, but they have been tested with **any** dataset.

The first command is common for every experiment: change the working directory to `experiments_pipeline` with

    cd experiments_pipeline

###### Full-Dataset Random Forest (Multi / Binary)

Use the parameter `--target` as "multi" for multi-classification, else "binary".

    python3 -m luigi --module pipeline FullDatasetRF --dataset-name "CIC-IDS2017" --target "multi" --tuning-min-samples-split "[2, 5, 10]" --tuning-min-samples-leaf "[1, 2, 4]" --tuning-iterations 10 --local-scheduler

###### Full-Dataset Supervised Gaussian Mixture Models

`--attack-only` is a flag used for filtering only attack data when true (by default true).

`--train-percentage` is an integer representing the percentage of the training data to consider (by default 100).

`--max-components` is the max number of mixture components that can be used globally to represent attack types: if `--tune-n-components` is false, the number is fixed, else it's the upper bound, i.e. tuning starting from 1 (by default 3).

`--covariance-type` is a string describing the type of covariance parameters to use, which must be "full" or "tied" or "diag" or "spherical" (by default "full").

`--reg-covar` is the non-negative regularization added to the diagonal of covariance (by default 1e-6).

`--tune-n-components` is a flag used to decide whether to tune the number of components globally in GMMs or not (by default true).

`--selection-metric` is a string describing which metric ("AIC", "f1_score" or "accuracy") to use on the validation set for tuning if `--tune-n-components` is true, else it's calculated once since the number of components is fixed (by default "AIC").

    python3 -m luigi --module pipeline FullDatasetSupervisedGMM --dataset-name "CIC-IDS2017" --attack-only true --train-percentage 100 --max-components 3 --covariance-type "full" --reg-covar 1e-6 --tune-n-components false --selection-metric "AIC" --local-scheduler

###### Continual Learning Supervised GMM

`--covariance-type` is a string describing the type of covariance parameters to use, which must be "full" or "tied" or "diag" or "spherical" (by default "full").

`--reg-covar` is the non-negative regularization added to the diagonal of covariance (by default 1e-6).

`--n-components` is the number of components globally in GMMs (by default 3).

`--permute-tasks` is a flag representing whether to use random permutations for tasks or not (by default false).

    python3 -m luigi --module pipeline ContinualSupervisedGMM --dataset-name "CIC-IDS2017" --covariance-type "full" --reg-covar 1e-6 --n-components 3 --permute-tasks false --local-scheduler

###### Continual Learning Bayesian GMM

`--covariance-type` is a string describing the type of covariance parameters to use, which must be "full" or "tied" or "diag" or "spherical" (by default "full").

`--reg-covar` is the non-negative regularization added to the diagonal of covariance (by default 1e-6).

`--n-components` is the number of components globally in GMMs (by default 3).

`--weight-concentration-prior-type` is the type of the weight concentration prior, which must be "dirichlet_process" or "dirichlet_distribution" (by default "dirichlet_process").

`--weight-concentration-prior` is the Dirichlet concentration of each component on the weight distribution (by default 0.01).

`--max-iter` is the number of EM iterations to perform (by default 100).

    python3 -m luigi --module pipeline ContinualBayesianGMM --dataset-name "CIC-IDS2017" --covariance-type "full" --reg-covar 1e-6 --n-components 3 --weight-concentration-prior-type "dirichlet_process" --weight-concentration-prior 0.01 --max-iter 100 --local-scheduler

###### Full-Dataset Neural Network

`--batch-size` is an integer representing the batch size used in the data loader (by default 128).

`--learning-rate` is the learning rate used in the optimizer (by default 0.001).

    python3 -m luigi --module pipeline FullDatasetNN --dataset-name "CIC-IDS2017" --batch-size 128 --learning-rate 0.001 --local-scheduler

###### Continual Learning Bayesian Neural Network (MLP, Laplace approximation)

`--batch-size` is an integer representing the batch size used in the data loader (by default 128).

`--learning-rate` is the learning rate used in the optimizer (by default 0.001).

`--lam` is the regularization strength (by default 1.0).

`--permute-tasks` is a flag representing whether to use random permutations for tasks or not (by default false).

    python3 -m luigi --module pipeline ContinualBNN --dataset-name "CIC-IDS2017" --batch-size 128 --learning-rate 0.001 --lam 1.0 --permute-tasks false --local-scheduler

###### Full-Dataset SVD Supervised Gaussian Mixture Models

`--n-components-SVD` is the number of SVD (Singular Value Decomposition) components used for dimensionality reduction (by default 30)

This Luigi task was created for experimenting with TON-IoT dataset in particular, so the default values refer to the best combination found for it.

    python3 -m luigi --module pipeline FullDatasetSVDGMM --dataset-name "TON-IoT" --attack-only true --train-percentage 100 --max-components 5 --covariance-type "full" --reg-covar 1e-2 --tune-n-components true --selection-metric "AIC" --n-components-SVD 30 --local-scheduler

###### FINAL SYSTEM - Continual Learning BNN + GMM

This is the final system made of two modules: BNN for binary classification, then the data classified as attack is passed to GMMs for multi-class classification.

It is based on the experiments **Continual Learning Bayesian Neural Network** and **Continual Learning Supervised GMM**.

    python3 -m luigi --module pipeline ContinualBNNPlusGMM --dataset-name "CIC-IDS2017" --batch-size 128 --learning-rate 0.001 --lam 1.0 --permute-tasks false --covariance-type "full" --reg-covar 1e-6 --n-components 3 --train-percentage 100 --local-scheduler

###### Full-Dataset Neural Network Multi

Multi version, which uses both benign and attack data.

    python3 -m luigi --module pipeline FullDatasetNNMulticlass --dataset-name "CIC-IDS2017" --batch-size 128 --learning-rate 0.001 --local-scheduler

###### FINAL SYSTEM FULL NORMALIZATION - Continual Learning BNN + GMM

Like the final system but normalization is performed using mean and std calculated on the whole training set, for comparison purposes.

    python3 -m luigi --module pipeline ContinualBNNPlusGMMFullNorm --dataset-name "CIC-IDS2017" --batch-size 128 --learning-rate 0.001 --lam 1.0 --permute-tasks false --covariance-type "full" --reg-covar 1e-6 --n-components 3 --local-scheduler