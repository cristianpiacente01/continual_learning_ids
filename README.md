# Master's thesis about CL for IDS

Repository containing pipelines for preparing data and experimenting for my Master's thesis about Continual Learning in the Intrusion Detection Systems domain.

Cristian Piacente 866020 @ University of Milan - Bicocca

## How to Run

### Setup Virtual Environment

First of all, it's necessary to create a Python virtual environment and to install all the dependencies.

#### Creating and activating

Using the repository **root directory** as the working directory, execute the following command to **create the virtual environment**:

    python -m venv venv

Once the virtual environment has been created, it will be saved in the filesystem, so it won't be necessary to recreate it every time.

Now, it's possible to **activate it** with

    source ./venv/Scripts/activate

Please note: if you are on Windows, remove `source` from the command.

#### Installing the dependencies

After the **first activation**, it's necessary to **install all the needed libraries** by executing

    pip install -r requirements.txt

Once the dependencies have been installed, it won't be necessary anymore for the future activations.
With the virtual environment activated, it's possible to execute the different parts of the project, described in the other sections.

#### Deactivating

To quit the virtual environment, it's sufficient to execute the command

    deactivate

### Pipeline (TODO)

#### Design (TODO)

##### DATA PREPARATION PIPELINE

**Inputs**: 

- dataset name (used for identifying the datasets folder, make sure you downloaded it from the [GDrive](https://drive.google.com/drive/folders/1auey1u1GrCB29wmuOXzskXgjSdlHo5f8?usp=sharing))
- file extension to consider (csv/parquet/txt, used to identify the correct files in the folder)
- int threshold parameter (see 2a. in "Steps", default=3000)
- parameters for preprocessing:
    - list parameter of features to drop (default=[])
    - list parameter of attack types to drop (default=[])

**Steps**:

0. Merge all data (train-val-test) into one file if not already done
1. Preprocessing, but no normalization yet (see "Preprocessing steps")
2. Split into two distributions: 
    2a. Top classes (including benign) where number of samples >= threshold parameter (e.g. 3000), i.e. **Closed-Set** multi-classification (known classes)
    2b. The rest where we set the labels to "unknown", i.e. **Out-of-Distribution** binary classification (0-day attacks) 
3. Split 2a. into tasks (batches), each one with the same number (based on the minority class from 2a.) of benign flows and 1 attack type flows
4. Split each task into train-val-test (60-20-20%)
5. Normalization per-task, using the train

###### Preprocessing steps

- automatic creation of columns attack, attack_type
- drop features in a list parameter
- drop constant features
- drop attack types in a list parameter
- data cleaning (drop duplicates, fix missing values, inf values, negative values)
- encoding for categorical features 
- casting types properly

##### EXPERIMENTING PIPELINE

**Inputs to be defined**, for sure the data preparation pipeline's outputs

**Steps**:

6. 
    a. Baseline modeling and evaluation on the full dataset (i.e., use the whole dataset as the only task, skipping 2. and 3.) \
    b. Modeling and evaluation on each task (simulate a real-time data stream), without catastrophic forgetting: \
        - Incrementally train (without storing the previous train sets) \
        - Predict on the union of all test sets up to the current task (Continual Learning while checking the ability of retaining past knowledge)
7. Use the OoD data for testing generalization capabilities


#### Execution (TODO)

##### DATA PREPARATION PIPELINE

With the virtual environment activated in the repository root, to execute the Data Preparation pipeline run the following commands:

 1.     cd data_prep_pipeline
 2.     python -m luigi --module pipeline MergeFiles --dataset-name "CIC-IDS2017" --input-file-extension ".parquet" --local-scheduler

 where the parameters are `--dataset-name`, `--input-file-extension`.