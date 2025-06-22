import luigi
import logging
import os
import pandas as pd
import numpy as np
import glob
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from laplace import Laplace
from torch.nn.utils import parameters_to_vector
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import TruncatedSVD
import random
from laplace.curvature.asdl import AsdlGGN

# Fix logging for Python 3.12.0
logging.root.handlers.clear()

# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('experiments-pipeline')

# Get configuration file
luigi.configuration.LuigiConfigParser.add_config_path("../luigi.cfg")
config = luigi.configuration.get_config()

# Dict that contains some parameters configurated in luigi.cfg
cfg = {
    'datasets_folder': config.get('DataPreparationPipeline', 'datasets_folder'),
    'preprocessed_rel_filename': config.get('DataPreparationPipeline', 'preprocessed_rel_filename'),
    'closed_set_rel_filename': config.get('DataPreparationPipeline', 'closed_set_rel_filename'),
    'ood_rel_filename': config.get('DataPreparationPipeline', 'ood_rel_filename'),
    'split_closed_set_rel_folder': config.get('DataPreparationPipeline', 'split_closed_set_rel_folder'),
    'tasks_rel_folder': config.get('DataPreparationPipeline', 'tasks_rel_folder'), 

    'models_folder': config.get('ExperimentsPipeline', 'models_folder'),
    'metrics_rel_filename': config.get('ExperimentsPipeline', 'metrics_rel_filename'),
    'nn_metrics_rel_filename': config.get('ExperimentsPipeline', 'nn_metrics_rel_filename'),
    'nn_multi_metrics_rel_filename': config.get('ExperimentsPipeline', 'nn_multi_metrics_rel_filename'),
    'continual_metrics_rel_filename': config.get('ExperimentsPipeline', 'continual_metrics_rel_filename'),
    'bnn_continual_metrics_rel_filename': config.get('ExperimentsPipeline', 'bnn_continual_metrics_rel_filename'),
    'bayesian_continual_metrics_rel_filename': config.get('ExperimentsPipeline', 'bayesian_continual_metrics_rel_filename'),
    'bnn_gmm_continual_metrics_rel_filename': config.get('ExperimentsPipeline', 'bnn_gmm_continual_metrics_rel_filename'),
    'full_norm_bnn_gmm_continual_metrics_rel_filename': config.get('ExperimentsPipeline', 'full_norm_bnn_gmm_continual_metrics_rel_filename'),
}

# Retrieve a relative path w.r.t the dataset name in the datasets folder
# suffix is the substring of the path after the dataset folder
def get_full_dataset_rel_path(dataset_name, suffix):
    return f'{cfg["datasets_folder"]}/{dataset_name}/{suffix}'

# Retrieve a relative path w.r.t the dataset name in the models folder
# suffix is the substring of the path after the model folder
def get_full_model_rel_path(dataset_name, suffix):
    return f'{cfg["models_folder"]}/{dataset_name}/{suffix}'

# Define fixed MLP architecture for BNN model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)  # Generalized output layer
        )

    def forward(self, x):
        return self.net(x)


class DirectoryTarget(luigi.Target):
    """
    Luigi Target for checking the existence of a directory
    """
    
    def __init__(self, path):
        self.path = path

    # Luigi's complete method override
    def complete(self):
        return os.path.isdir(self.path)

    # Luigi's complete method override
    def exists(self):
        return os.path.isdir(self.path)


class FullDatasetRF(luigi.Task):
    """
    Random Forest binary or multi (depending on a parameter) classification on the full dataset
    """

    dataset_name = luigi.Parameter()
    target = luigi.Parameter(default="binary") # binary/multi
    tuning_min_samples_split = luigi.ListParameter(default=(2, 5, 10)) # Used for tuning
    tuning_min_samples_leaf = luigi.ListParameter(default=(1, 2, 4)) # Used for tuning
    tuning_iterations = luigi.IntParameter(default=3) # Used for tuning

    def requires(self):
        # Train, validation and test are needed 
        class FakeTask(luigi.Task):
            def output(_):
                return DirectoryTarget(get_full_dataset_rel_path(self.dataset_name,
                                                                 cfg['split_closed_set_rel_folder']))

        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Flag that is True if multi-classification, False if binary
        multi = self.target.lower().startswith("multi")

        # Classification Type
        classification_type = "MULTI" if multi else "BINARY"

        logger.info(f'Type of classification: {classification_type}')

        # Load the train dataset
        train_df = pd.read_csv(os.path.join(self.input().path, 'train.csv'))

        logger.info(f'Loaded train set from {self.input().path}/train.csv')

        # Load the validation dataset
        val_df = pd.read_csv(os.path.join(self.input().path, 'val.csv'))

        logger.info(f'Loaded validation set from {self.input().path}/val.csv')

        # Load the test dataset
        test_df = pd.read_csv(os.path.join(self.input().path, 'test.csv'))

        logger.info(f'Loaded test set from {self.input().path}/test.csv')

        # Extract features
        X_train = train_df.drop(columns=["attack", "attack_type"])
        X_val = val_df.drop(columns=["attack", "attack_type"])
        X_test = test_df.drop(columns=["attack", "attack_type"])

        # Extract labels (Target for classification)
        y_train = train_df["attack_type" if multi else "attack"]
        y_val = val_df["attack_type" if multi else "attack"]
        y_test = test_df["attack_type" if multi else "attack"]
        
        # Identify columns to normalize
        columns_to_normalize = list(X_train.select_dtypes(include=['float', 'int']).columns)

        logger.info(f'Applying Z-Score normalization on columns {columns_to_normalize}')

        # Z-Score normalization using train, for each chosen column individually
        mean = X_train[columns_to_normalize].mean()
        std = X_train[columns_to_normalize].std()

        X_train[columns_to_normalize] = (X_train[columns_to_normalize] - mean) / std
        X_val[columns_to_normalize] = (X_val[columns_to_normalize] - mean) / std
        X_test[columns_to_normalize] = (X_test[columns_to_normalize] - mean) / std

        logger.info(f'Normalized columns')

        ##### --- HYPERPARAMETER TUNING --- #####

        # Define hyperparameter search space
        param_distributions = {
            "n_estimators": [500], # From the paper Machine_Learning_based_Intrusion_Detection_Systems for_IoT_Applications.pdf
            "max_depth": [26], # From the paper Machine_Learning_based_Intrusion_Detection_Systems for_IoT_Applications.pdf
            "max_features": ["sqrt", "log2", None], # All possibilities
            "min_samples_split": list(self.tuning_min_samples_split), # Input parameter
            "min_samples_leaf": list(self.tuning_min_samples_leaf), # Input parameter
        }

        # Randomized search for hyperparameter tuning
        best_model = None
        best_score = -np.inf
        best_params = None

        logger.info(f'Tuning the model')

        for params in ParameterSampler(param_distributions, n_iter=self.tuning_iterations, random_state=42):
            # Train a Random Forest with the current parameters on the training set
            model = RandomForestClassifier(random_state=42, **params)
            model.fit(X_train, y_train)

            # Evaluate on the validation set
            val_score = model.score(X_val, y_val)

            # Update best model if current model performs better on validation
            if val_score > best_score:
                best_model = model
                best_score = val_score
                best_params = params

        logger.info(f'Best hyperparameters: {best_params}')

        ##### --- EVALUATION ON TEST SET --- #####

        # Evaluate the best model on the test set
        y_test_pred = best_model.predict(X_test)

        # Average parameter for scores
        average = "macro" if multi else "binary"

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average=average)
        recall = recall_score(y_test, y_test_pred, average=average)
        f1 = f1_score(y_test, y_test_pred, average=average)

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score: {f1}')
        # If binary then calculate AUROC
        if average == "binary":
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_test_proba)
            logger.info(f'-> AUROC: {auc_roc}')
            f1_weighted = None # No F1 Score Weighted for binary
        else:
            auc_roc = None # No AUROC for multi-classification
            f1_weighted = f1_score(y_test, y_test_pred, average="weighted")
            logger.info(f'-> F1-Score Weighted: {f1_weighted}')

        ##### --- SAVE THE MODEL AND METRICS --- #####

        # Parametric model name
        model_name = f'Full-Dataset_RF_{classification_type}'

        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, f'{model_name}_model.pkl')
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        # Save .pkl file
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)

        logger.info(f'Saved the model to {model_file}')

        # Create a DataFrame for the metrics
        metrics_df = pd.DataFrame([{
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f1_score_weighted": f1_weighted if f1_weighted is not None else "N/A (Binary)",
            "roc_auc": auc_roc if auc_roc is not None else "N/A (Multi)",
            "best_hyperparameters": str(best_params),
        }])

        # Append the data to metrics.csv
        with open(metrics_csv, 'a') as f: 
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')

        logger.info(f'Finished task {self.__class__.__name__}')



class FullDatasetSupervisedGMM(luigi.Task):
    """
    Supervised Gaussian Mixture Model (GMM) for multi-class classification
    """

    dataset_name = luigi.Parameter()

    # Filter only attack data
    attack_only = luigi.BoolParameter(default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING) 

    # Percentage of training data to use
    train_percentage = luigi.IntParameter(default=100) 

    # Max number of components that can be used globally to represent attack types
    # If tune_n_components is false, the number is fixed, else it's the upper bound (tuning starting from 1)
    max_components = luigi.IntParameter(default=3)

    # Type of covariance matrix
    covariance_type = luigi.Parameter(default="full")

    # Regularization for covariance matrix
    reg_covar = luigi.FloatParameter(default=1e-6)

    # Decide whether to tune the number of components globally in GMMs or not, by default yes
    tune_n_components = luigi.BoolParameter(default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    # Metric (AIC, f1_score or accuracy) used for tuning on validation set if tune_n_components is true
    # If not tuning, it's calculated once since the number of components is fixed
    selection_metric = luigi.Parameter(default="AIC")

    def requires(self):
        # Train, validation and test are needed 
        class FakeTask(luigi.Task):
            def output(_):
                return DirectoryTarget(get_full_dataset_rel_path(self.dataset_name,
                                                                 cfg['split_closed_set_rel_folder']))

        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info('====== PARAMETERS ======')
        logger.info(f'attack_only={self.attack_only}')
        logger.info(f'train_percentage={self.train_percentage}')
        logger.info(f'max_components={self.max_components}')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'tune_n_components={self.tune_n_components}')
        logger.info(f'selection_metric={self.selection_metric}')
        logger.info('========================')

        # Load the train dataset
        train_df = pd.read_csv(os.path.join(self.input().path, 'train.csv'))

        logger.info(f'Loaded train set from {self.input().path}/train.csv')

        # Load the validation dataset
        val_df = pd.read_csv(os.path.join(self.input().path, 'val.csv'))

        logger.info(f'Loaded validation set from {self.input().path}/val.csv')

        # Load the test dataset
        test_df = pd.read_csv(os.path.join(self.input().path, 'test.csv'))

        logger.info(f'Loaded test set from {self.input().path}/test.csv')

        if self.attack_only:
            train_df = train_df[train_df['attack'] == True]
            val_df = val_df[val_df['attack'] == True]
            test_df = test_df[test_df['attack'] == True]
            logger.info(f'Filtered only attack data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

        # Apply percentage-based sampling on training set
        if self.train_percentage > 0 and self.train_percentage < 100:
            train_df, _ = train_test_split(train_df, train_size=self.train_percentage / 100, stratify=train_df["attack"], random_state=42)
            logger.info(f'Using only {self.train_percentage}% of training data: {len(train_df)} samples')

        # Extract features
        X_train = train_df.drop(columns=["attack", "attack_type"])
        X_val = val_df.drop(columns=["attack", "attack_type"])
        X_test = test_df.drop(columns=["attack", "attack_type"])

        # Extract labels
        y_train = train_df["attack_type"]
        y_val = val_df["attack_type"]
        y_test = test_df["attack_type"]

        # Encode attack types to numeric values
        attack_types = sorted(y_train.unique())
        attack_mapping = {attack: i for i, attack in enumerate(attack_types)}
        y_train = y_train.map(attack_mapping)
        y_val = y_val.map(attack_mapping)
        y_test = y_test.map(attack_mapping)

        logger.info(f'Mapped attack types to numerical labels: {attack_mapping}')

        # Identify columns to normalize: numeric columns which are NOT CONSTANT in the new train
        columns_to_normalize = [numeric_column
                                for numeric_column in list(X_train.select_dtypes(include=['float', 'int']).columns)
                                if X_train[numeric_column].nunique() > 1]

        logger.info(f'Applying Z-Score normalization on columns {columns_to_normalize}')

        # Z-Score normalization using train, for each chosen column individually
        mean = X_train[columns_to_normalize].mean()
        std = X_train[columns_to_normalize].std()

        X_train[columns_to_normalize] = (X_train[columns_to_normalize] - mean) / std
        X_val[columns_to_normalize] = (X_val[columns_to_normalize] - mean) / std
        X_test[columns_to_normalize] = (X_test[columns_to_normalize] - mean) / std

        logger.info(f'Normalized columns')

        ##### --- GLOBAL TUNING LOOP --- #####
        best_class_gmms = None       # Dictionary of class_idx -> best GMM
        best_score = -np.inf         # Best validation score observed so far
        best_n_components = None     # Number of components corresponding to best_score

        # If tuning try number of components from 1 to max_components, else just one iteration (max_components itself)
        for n_components in range(1, self.max_components + 1) if self.tune_n_components else [self.max_components]:
            
            logger.info(f'Trying n_components = {n_components}')

            class_gmms = {}

            # Train one GMM per class
            for attack_type in attack_types:
                class_idx = attack_mapping[attack_type]
                X_train_class = X_train[y_train == class_idx]

                gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar,
                                      random_state=np.random.default_rng().integers(0, 2**32 - 1)) # random seed
                gmm.fit(X_train_class)
                class_gmms[class_idx] = gmm

            # Use all GMMs to predict on validation set
            log_likelihoods = np.zeros((X_val.shape[0], len(class_gmms)))
            for class_idx, gmm in class_gmms.items():
                log_likelihoods[:, class_idx] = gmm.score_samples(X_val)
            y_val_pred = np.argmax(log_likelihoods, axis=1)

            # Compute validation score based on selection metric
            if self.selection_metric.lower() == "accuracy":
                score = accuracy_score(y_val, y_val_pred)
            elif self.selection_metric.lower() == "f1_score":
                score = f1_score(y_val, y_val_pred, average="macro")
            else: # == "aic":
                # We want to minimize AIC not maximize, so we use a negative sign
                score = -sum(class_gmms[class_idx].aic(X_val[y_val == class_idx])
                            for _, class_idx in attack_mapping.items())

            if self.selection_metric.lower() == "aic":
                logger.info(f'n_components={n_components} -> Validation AIC = {-score} (minimization)')
            else:
                logger.info(f'n_components={n_components} -> Validation {self.selection_metric} = {score} (maximization)')

            # Update best model if better than previous ones
            if score > best_score:
                best_score = score
                best_class_gmms = class_gmms
                best_n_components = n_components
                logger.info(f'Best n_components so far: {n_components}')

        ##### --- PREDICT ON TEST SET --- #####
        log_likelihoods = np.zeros((X_test.shape[0], len(best_class_gmms)))
        for class_idx, gmm in best_class_gmms.items():
            log_likelihoods[:, class_idx] = gmm.score_samples(X_test)
        y_test_pred = np.argmax(log_likelihoods, axis=1)

        # Compute final test metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="macro")
        recall = recall_score(y_test, y_test_pred, average="macro")
        f1 = f1_score(y_test, y_test_pred, average="macro")
        f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score (Macro): {f1}')
        logger.info(f'-> F1-Score (Weighted): {f1_weighted}')
        

        ##### --- SAVE MODEL AND METRICS --- #####
        model_name = f'Full-Dataset_SupervisedGMM{"_AttackOnly" if self.attack_only else ""}_{self.train_percentage}%' + \
                    f'_{"Tune" if self.tune_n_components else "Fixed"}-{best_n_components}_{self.selection_metric}'

        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, f'{model_name}.pkl')
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        # Save model as pickle
        with open(model_file, 'wb') as f:
            pickle.dump(best_class_gmms, f)

        # Save evaluation metrics to CSV
        metrics_df = pd.DataFrame([{
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f1_score_weighted": f1_weighted,
            "roc_auc": "N/A (Multi)",
            "best_hyperparameters": str({
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                "selection_metric": self.selection_metric,
                "tune_components": self.tune_n_components,
                "n_components": best_n_components
            }),
        }])

        with open(metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class ContinualSupervisedGMM(luigi.Task):
    """
    Supervised GMM for Continual Learning multi-class classification
    """

    dataset_name = luigi.Parameter()

    # Type of covariance matrix
    covariance_type = luigi.Parameter(default="full")

    # Regularization for covariance matrix
    reg_covar = luigi.FloatParameter(default=1e-6)

    # Number of components used to represent each attack type, this number is fixed
    n_components = luigi.IntParameter(default=3)

    # Flag to decide whether to permute tasks or not
    permute_tasks = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def requires(self):
        # The "split-dataset" (val.csv and test.csv) and "tasks" folders are needed
        class FakeTask(luigi.Task):
            def output(_):
                return {'split-dataset': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                 cfg['split_closed_set_rel_folder'])),
                        'tasks': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                           cfg['tasks_rel_folder']))}
            
        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f'====== PARAMETERS (DATASET {self.dataset_name}) ======')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'n_components={self.n_components}')
        logger.info('========================')

        split_path = self.input()['split-dataset'].path
        tasks_path = self.input()['tasks'].path
        
        # Retrieve val.csv and test.csv from the split-dataset folder, then the csv files from the tasks folder
        val_df = pd.read_csv(os.path.join(split_path, "val.csv"))
        test_df = pd.read_csv(os.path.join(split_path, "test.csv"))
        task_files = sorted(glob.glob(os.path.join(tasks_path, "task_*.csv")))
        if self.permute_tasks:
            random.shuffle(task_files)

        label_map = {}
        next_class_index = 0
        gmms = {}
        metrics = []

        class_sample_counts = {}  # dict for prior computation
        first_task_mean = None
        first_task_std = None

        for i, task_file in enumerate(task_files):
            task_df = pd.read_csv(task_file)

            current_attacks = set(task_df["attack_type"].unique()) - {"benign", "Benign", "normal", "Normal"} 
            assert len(current_attacks) == 1, f"Task file {task_file} must contain exactly 1 attack type"
            current_attack = current_attacks.pop()

            # Assign index if first time seeing this attack type
            if current_attack not in label_map:
                label_map[current_attack] = next_class_index
                next_class_index += 1

            class_idx = label_map[current_attack]
            
            # --- TRAINING SET FROM THE TASK CONSIDERING JUST THE CURRENT ATTACK, NO BENIGN TRAFFIC ---
            task_df = task_df[task_df["attack_type"] == current_attack]
            X_train = task_df.drop(columns=["attack", "attack_type"])
            #y_train = np.full(len(X_train), class_idx)

            # --- VALIDATION SET WITH JUST THE CURRENT ATTACK ---
            val_task_df = val_df[val_df["attack_type"] == current_attack]
            X_val = val_task_df.drop(columns=["attack", "attack_type"])
            #y_val = np.full(len(X_val), class_idx)

            # --- CUMULATIVE TEST SET WITH THE SEEN ATTACKS UP TO NOW ---
            seen_attack_types = set(label_map.keys())
            test_seen_df = test_df[test_df["attack_type"].isin(seen_attack_types)]
            X_test = test_seen_df.drop(columns=["attack", "attack_type"])
            y_test = test_seen_df["attack_type"].map(label_map)

            # For tracking test metrics on the current task and previous tasks
            test_current_df = test_df[test_df["attack_type"] == current_attack]
            test_previous_df = test_df[test_df["attack_type"].isin(seen_attack_types - {current_attack})] \
                                if i > 0 else None

            # Identify columns to normalize: numeric columns, keep also constant ones since they may change among different tasks!
            columns_to_normalize = [numeric_column
                                    for numeric_column in list(X_train.select_dtypes(include=['float', 'int']).columns)]

            # Retrieve mean and std if it's the first task
            if first_task_mean is None:
                first_task_mean = X_train[columns_to_normalize].mean()
                first_task_std = X_train[columns_to_normalize].std()
            
            # --- Z-SCORE NORMALIZATION USING MEAN & STD FROM 1st TASK ---
            for df in [X_train, X_val, X_test, test_current_df, test_previous_df]:
                if df is None or df.empty:
                    continue
                for col in columns_to_normalize:
                    mean = first_task_mean[col]
                    std = first_task_std[col]
                    # Avoid NaN or something that isn't ok for training a GMM
                    if pd.isna(mean) or mean < 1e-8 or pd.isna(std) or std < 1e-8:
                        df[col] = 0.0
                    else:
                        df[col] = (df[col] - mean) / std

            # Train new GMM for current attack
            gmm = GaussianMixture(n_components=self.n_components,
                                   covariance_type=self.covariance_type,
                                   reg_covar=self.reg_covar,
                                   random_state=42)
            gmm.fit(X_train)
            gmms[class_idx] = gmm

            # AIC on validation
            aic = gmm.aic(X_val)
            logger.info(f'Task {i + 1} | Attack: {current_attack} | AIC (val): {aic:.2f}')

            # Update class sample count (manual check)
            if class_idx not in class_sample_counts:
                class_sample_counts[class_idx] = 0
            class_sample_counts[class_idx] += len(X_train)
            total_samples_seen = sum(class_sample_counts.values())

            # === TEST EVALUATIONS - HELPER FUNCTION ===
            def eval_metrics(X, y, label):
                # Compute log-likelihood + log-prior for each class
                log_likelihoods = np.zeros((X.shape[0], len(gmms)))
                for gmm_class_idx, gmm_model in gmms.items():
                    log_likelihood = gmm_model.score_samples(X)
                    prior_prob = class_sample_counts[gmm_class_idx] / total_samples_seen
                    log_prior = np.log(prior_prob)
                    log_likelihoods[:, gmm_class_idx] = log_likelihood + log_prior
                
                y_pred = np.argmax(log_likelihoods, axis=1)
                acc = accuracy_score(y, y_pred)
                logger.info(f'[TASK {i+1}] {label} accuracy: {acc:.4f}')
                return {
                    'accuracy': acc,
                    'f1_macro': f1_score(y, y_pred, average="macro", zero_division=0),
                    'f1_weighted': f1_score(y, y_pred, average="weighted", zero_division=0)
                }
            
            # Cumulative, current and previous tasks metrics
            metrics_cumulative = eval_metrics(X_test, y_test, "CUMULATIVE")
            metrics_current = eval_metrics(test_current_df.drop(columns=["attack", "attack_type"]), 
                                           test_current_df["attack_type"].map(label_map), 
                                           'CURRENT')
            metrics_previous = {metric: None for metric in ['accuracy', 'f1_macro', 'f1_weighted']} if i == 0 \
                         else eval_metrics(test_previous_df.drop(columns=["attack", "attack_type"]), 
                                           test_previous_df['attack_type'].map(label_map), 
                                           'PREVIOUS')
            
            metrics.append({
                "dataset": self.dataset_name,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                "permute_tasks": self.permute_tasks,
                "task": i + 1,
                "attack": current_attack,
                **{f'cumulative_{k}': v for k, v in metrics_cumulative.items()},
                **{f'current_{k}': v for k, v in metrics_current.items()},
                **{f'previous_{k}': v for k, v in metrics_previous.items()},
                "aic_val": aic
            })

        # --- SAVE MODEL AND METRICS ---
        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, 'Continual_Learning_SupervisedGMM.pkl')
        continual_metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['continual_metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(continual_metrics_csv), exist_ok=True)

        # Save model as pickle
        with open(model_file, 'wb') as f:
            pickle.dump(gmms, f)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        logger.info('Metrics:')
        logger.info(f'\n{metrics_df}')

        with open(continual_metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {continual_metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class FullDatasetNN(luigi.Task):
    """
    Full-dataset binary classification using a NN
    """

    dataset_name = luigi.Parameter()

    # Batch size
    batch_size = luigi.IntParameter(default=128)

    # Learning rate 
    learning_rate = luigi.FloatParameter(default=0.001)

    def requires(self):
        # Train, validation and test are needed 
        class FakeTask(luigi.Task):
            def output(_):
                return DirectoryTarget(get_full_dataset_rel_path(self.dataset_name,
                                                                 cfg['split_closed_set_rel_folder']))

        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info('====== PARAMETERS ======')
        logger.info(f'batch_size={self.batch_size}')
        logger.info(f'learning_rate={self.learning_rate}')
        logger.info('========================')

        # Set PyTorch random seed
        torch.manual_seed(np.random.default_rng().integers(0, 2**32 - 1))

        # Load the train dataset
        train_df = pd.read_csv(os.path.join(self.input().path, 'train.csv'))

        logger.info(f'Loaded train set from {self.input().path}/train.csv')

        # Load the validation dataset
        val_df = pd.read_csv(os.path.join(self.input().path, 'val.csv'))

        logger.info(f'Loaded validation set from {self.input().path}/val.csv')

        # Load the test dataset
        test_df = pd.read_csv(os.path.join(self.input().path, 'test.csv'))

        logger.info(f'Loaded test set from {self.input().path}/test.csv')

        # Extract features
        X_train = train_df.drop(columns=["attack", "attack_type"])
        X_val = val_df.drop(columns=["attack", "attack_type"])
        X_test = test_df.drop(columns=["attack", "attack_type"])

        # Extract labels
        y_train = train_df["attack"].astype(int).values
        y_val = val_df["attack"].astype(int).values
        y_test = test_df["attack"].astype(int).values

        # Identify columns to normalize
        columns_to_normalize = list(X_train.select_dtypes(include=['float', 'int']).columns)

        logger.info(f'Applying Z-score normalization on columns: {columns_to_normalize}')

        # Z-Score normalization using train, for each chosen column individually
        mean = X_train[columns_to_normalize].mean()
        std = X_train[columns_to_normalize].std()

        X_train[columns_to_normalize] = (X_train[columns_to_normalize] - mean) / std
        X_val[columns_to_normalize] = (X_val[columns_to_normalize] - mean) / std
        X_test[columns_to_normalize] = (X_test[columns_to_normalize] - mean) / std

        X_train_t = torch.tensor(X_train.values.astype(np.float32))
        X_val_t = torch.tensor(X_val.values.astype(np.float32))
        X_test_t = torch.tensor(X_test.values.astype(np.float32))

        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=self.batch_size, shuffle=True)

        # Initialize model and optimizer
        model = MLP(X_train_t.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(10):
            model.train()
            running_loss = 0
            correct_train = 0
            total_train = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

                pred = out.argmax(dim=1)
                correct_train += (pred == yb).sum().item()
                total_train += yb.size(0)

            avg_train_loss = running_loss / total_train
            train_acc = correct_train / total_train

            # Evaluate on validation
            model.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    pred = out.argmax(dim=1)
                    correct_val += (pred == yb).sum().item()
                    total_val += yb.size(0)

            avg_val_loss = val_loss / total_val
            val_acc = correct_val / total_val

            logger.info(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')

        logger.info(f'Training completed')

        logits_val = model(X_val_t).detach()
        probs_val = torch.softmax(logits_val, dim=1).numpy()
        y_pred_val = np.argmax(probs_val, axis=1)
        val_acc_post_laplace = accuracy_score(y_val, y_pred_val)
        logger.info(f'Validation accuracy: {val_acc_post_laplace:.4f}')

        # Inference
        model.eval()
        logits = model(X_test_t).detach()
        probs = torch.softmax(logits, dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        auc_roc = roc_auc_score(y_test, probs[:, 1])

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score (Macro): {f1}')
        logger.info(f'-> F1-Score (Weighted): {f1_weighted}')
        logger.info(f'-> AUROC: {auc_roc}')
        
        # Save model and metrics
        model_name = "Full-Dataset_NN_Binary"
        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, f'{model_name}_weights.pt')
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['nn_metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), model_file)

        # Save evaluation metrics to CSV
        metrics_df = pd.DataFrame([{
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f1_score_weighted": f1_weighted,
            "roc_auc": auc_roc,
            "best_hyperparameters": str({
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate
            })
        }])

        with open(metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class ContinualBNN(luigi.Task):
    """
    Continual Learning Binary Classification using BNN with Laplace approximation (Laplace Redux).
    Each task introduces a new attack type, but model predicts only binary: benign (0) vs attack (1).
    Posterior from previous task is used as prior regularizer during current training.
    """

    dataset_name = luigi.Parameter()

    # Batch size
    batch_size = luigi.IntParameter(default=128)

    # Learning rate 
    learning_rate = luigi.FloatParameter(default=0.001)

    # Regularization strength
    lam = luigi.FloatParameter(default=1.0)

    # Flag to decide whether to permute tasks or not
    permute_tasks = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def requires(self):
        # The "split-dataset" (val.csv and test.csv) and "tasks" folders are needed
        class FakeTask(luigi.Task):
            def output(_):
                return {'split-dataset': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                 cfg['split_closed_set_rel_folder'])),
                        'tasks': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                           cfg['tasks_rel_folder']))}
            
        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f'====== PARAMETERS (DATASET {self.dataset_name}) ======')
        logger.info(f'batch_size={self.batch_size}')
        logger.info(f'learning_rate={self.learning_rate}')
        logger.info(f'lam={self.lam}')
        logger.info(f'permute_tasks={self.permute_tasks}')
        logger.info('========================')

        split_path = self.input()['split-dataset'].path
        tasks_path = self.input()['tasks'].path

        # Retrieve val.csv and test.csv from the split-dataset folder, then the csv files from the tasks folder
        val_df = pd.read_csv(os.path.join(split_path, 'val.csv'))
        test_df = pd.read_csv(os.path.join(split_path, 'test.csv'))
        task_files = sorted(glob.glob(os.path.join(tasks_path, 'task_*.csv')))
        if self.permute_tasks:
            random.shuffle(task_files)

        model = None           # Shared model across tasks
        laplace = None         # Stores Laplace posterior from previous task
        metrics = []           # Collect evaluation metrics
        seen_attacks = []      # Track cumulative test attacks

        # Reuse first task stats for normalization (as per CL setup)
        first_task_mean = None
        first_task_std = None  

        # === MAIN CONTINUAL LEARNING LOOP (one task = one attack) ===
        for i, task_file in enumerate(task_files):
            task_df = pd.read_csv(task_file)

            # --- Get current attack class ---
            current_attacks = set(task_df["attack_type"].unique()) - {"benign", "Benign", "normal", "Normal"} 
            assert len(current_attacks) == 1, f"Task file {task_file} must contain exactly 1 attack type"
            current_attack = current_attacks.pop()
            seen_attacks.append(current_attack)

            # Train = benign + current attack
            X_train = task_df.drop(columns=["attack", "attack_type"])
            y_train = task_df['attack'].astype(int).values

            # Val = benign + current attack
            val_task_df = val_df[val_df['attack_type'].isin([current_attack, 'Benign', 'benign', 'normal', 'Normal'])]
            X_val = val_task_df.drop(columns=["attack", "attack_type"])
            y_val = val_task_df['attack'].astype(int).values

            # Test = benign + all seen attack types (label = 1), benign = 0
            test_seen_df = test_df[test_df['attack_type'].isin(seen_attacks + ['benign', 'Benign', 'normal', 'Normal'])]
            X_test = test_seen_df.drop(columns=["attack", "attack_type"])
            y_test = test_seen_df['attack'].astype(int).values

            # For tracking test metrics on the current task and previous tasks
            test_current_df = test_df[test_df['attack_type'].isin([current_attack, "benign", "Benign", "normal", "Normal"])]
            test_previous_df = test_df[test_df['attack_type'].isin(seen_attacks[:-1] + ["benign", "Benign", "normal", "Normal"])] \
                                if i > 0 else None

            # Identify columns to normalize: numeric columns, keep also constant ones since they may change among different tasks!
            columns_to_normalize = [numeric_column
                                    for numeric_column in list(X_train.select_dtypes(include=['float', 'int']).columns)]
            
            # Retrieve mean and std if it's the first task
            if first_task_mean is None:
                first_task_mean = X_train[columns_to_normalize].mean()
                first_task_std = X_train[columns_to_normalize].std()

            # --- Z-SCORE NORMALIZATION USING MEAN & STD FROM 1st TASK ---
            for df in [X_train, X_val, X_test, test_current_df, test_previous_df]:
                if df is None or df.empty:
                    continue
                for col in columns_to_normalize:
                    mean = first_task_mean[col]
                    std = first_task_std[col]
                    # Avoid NaN or something that isn't ok
                    if pd.isna(mean) or mean < 1e-8 or pd.isna(std) or std < 1e-8:
                        df[col] = 0.0
                    else:
                        df[col] = (df[col] - mean) / std

            # --- Convert to Torch tensors ---
            X_train_t = torch.tensor(X_train.values.astype(np.float32))
            X_val_t = torch.tensor(X_val.values.astype(np.float32))
            X_test_t = torch.tensor(X_test.values.astype(np.float32))
            y_train_t = torch.tensor(y_train, dtype=torch.long)
            y_val_t = torch.tensor(y_val, dtype=torch.long)
            y_test_t = torch.tensor(y_test, dtype=torch.long)

            train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=self.batch_size, shuffle=True)

            # === MODEL INITIALIZATION (first task only) ===
            if model is None:
                model = MLP(X_train_t.shape[1])

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            # === TRAINING LOOP ===
            for epoch in range(10):
                model.train()
                running_loss = 0
                correct_train = 0
                total_train = 0

                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)

                    # Add Laplace Redux regularization from previous posterior
                    if laplace is not None:
                        theta = parameters_to_vector(model.parameters())
                        log_prior = laplace.log_prob(theta) / len(xb)  # divide by batch size
                        loss = loss - self.lam * log_prior

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * xb.size(0)

                    pred = out.argmax(dim=1)
                    correct_train += (pred == yb).sum().item()
                    total_train += yb.size(0)

                avg_train_loss = running_loss / total_train
                train_acc = correct_train / total_train

                # === Validation accuracy (no Laplace) ===
                model.eval()
                correct_val = 0
                total_val = 0
                val_loss = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        out = model(xb)
                        loss = loss_fn(out, yb)
                        val_loss += loss.item() * xb.size(0)
                        pred = out.argmax(dim=1)
                        correct_val += (pred == yb).sum().item()
                        total_val += yb.size(0)

                avg_val_loss = val_loss / total_val
                val_acc = correct_val / total_val
                logger.info(f'[TASK {i + 1}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')

            # === FIT LAPLACE REDUX ===
            logger.info(f'[TASK {i + 1}] Training completed. Fitting Laplace approximation...')
            laplace = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='kron', backend=AsdlGGN)
            laplace.fit(train_loader)
            laplace.prior_mean = laplace.mean.clone()
            laplace.optimize_prior_precision(init_prior_prec=laplace.prior_precision)
            logger.info(f'[TASK {i + 1}] Laplace fitted')

            # === POSTERIOR-BASED VALIDATION & TEST ===
            logits_val_list = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    logits_batch = model(xb).detach()
                    logits_val_list.append(logits_batch)

            logits_val = torch.cat(logits_val_list, dim=0)
            probs_val = torch.softmax(logits_val, dim=1).numpy()
            y_pred_val = np.argmax(probs_val, axis=1)
            val_acc_post_laplace = accuracy_score(y_val, y_pred_val)
            logger.info(f'Validation accuracy: {val_acc_post_laplace:.4f}')

            # === TEST EVALUATIONS - HELPER FUNCTION ===
            def eval_metrics(loader, y, label):
                logits_list = []
                with torch.no_grad():
                    for xb, _ in loader:
                        logits_batch = model(xb).detach()
                        logits_list.append(logits_batch)
            
                logits = torch.cat(logits_list, dim=0)
                probs = torch.softmax(logits, dim=1).numpy()
                y_pred = np.argmax(probs, axis=1)
                acc = accuracy_score(y, y_pred)
                logger.info(f'[TASK {i+1}] {label} accuracy: {acc:.4f}')
                return {
                    'accuracy': acc,
                    'f1_macro': f1_score(y, y_pred, average="macro", zero_division=0),
                    'f1_weighted': f1_score(y, y_pred, average="weighted", zero_division=0),
                    'roc_auc': roc_auc_score(y, probs[:, 1])
                }
            
            # Test data loaders
            test_loader_all = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=self.batch_size)
            test_loader_current = DataLoader(TensorDataset(
                torch.tensor(test_current_df.drop(columns=["attack", "attack_type"]).values.astype(np.float32)),
                torch.tensor(test_current_df['attack'].astype(int).values)
            ), batch_size=self.batch_size)

            test_loader_prev = None
            if i > 0:
                test_loader_prev = DataLoader(TensorDataset(
                    torch.tensor(test_previous_df.drop(columns=["attack", "attack_type"]).values.astype(np.float32)),
                    torch.tensor(test_previous_df['attack'].astype(int).values)
                ), batch_size=self.batch_size)

            # Cumulative, current and previous tasks metrics
            metrics_cumulative = eval_metrics(test_loader_all, y_test, 'CUMULATIVE')
            metrics_current = eval_metrics(test_loader_current, test_current_df['attack'].astype(int).values, 'CURRENT')
            metrics_previous = {metric: None for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'roc_auc']} if i == 0 \
                         else eval_metrics(test_loader_prev, test_previous_df['attack'].astype(int).values, 'PREVIOUS')

            logger.info(f'Task {i + 1} -> Accuracy: {metrics_cumulative["accuracy"]:.4f} | F1 (Macro): {metrics_cumulative["f1_macro"]:.4f} | F1 (Weighted): {metrics_cumulative["f1_weighted"]:.4f}')

            metrics.append({
                'dataset': self.dataset_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'lambda': self.lam,
                'permute_tasks': self.permute_tasks,
                'task': i + 1,
                'attack': current_attack,
                **{f'cumulative_{k}': v for k, v in metrics_cumulative.items()},
                **{f'current_{k}': v for k, v in metrics_current.items()},
                **{f'previous_{k}': v for k, v in metrics_previous.items()}
            })

        # --- SAVE MODEL AND METRICS ---
        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, 'Continual_Learning_BNN_weights.pt')
        continual_metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['bnn_continual_metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(continual_metrics_csv), exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), model_file)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        logger.info('Metrics:')
        logger.info(f'\n{metrics_df}')

        with open(continual_metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {continual_metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class ContinualBayesianGMM(luigi.Task):
    """
    Continual Bayesian GMM for Continual Learning multi-class classification
    """

    dataset_name = luigi.Parameter()

    # Type of covariance matrix
    covariance_type = luigi.Parameter(default="full")

    # Regularization for covariance matrix
    reg_covar = luigi.FloatParameter(default=1e-6)

    # Number of components used to represent each attack type (BayesianGaussianMixture can infer not to use all)
    n_components = luigi.IntParameter(default=3)

    # Type of the weight concentration prior
    weight_concentration_prior_type = luigi.Parameter(default="dirichlet_process")  # or "dirichlet_distribution"

    # Dirichlet concentration of each component on the weight distribution (Dirichlet), i.e. gamma in the literature
    weight_concentration_prior = luigi.FloatParameter(default=0.01)

    # Number of EM iterations to perform
    max_iter = luigi.IntParameter(default=100)


    def requires(self):
        # The "split-dataset" (val.csv and test.csv) and "tasks" folders are needed
        class FakeTask(luigi.Task):
            def output(_):
                return {'split-dataset': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                 cfg['split_closed_set_rel_folder'])),
                        'tasks': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                           cfg['tasks_rel_folder']))}
            
        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f'====== PARAMETERS (DATASET {self.dataset_name}) ======')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'n_components={self.n_components}')
        logger.info(f'weight_concentration_prior_type={self.weight_concentration_prior_type}')
        logger.info(f'weight_concentration_prior={self.weight_concentration_prior}')
        logger.info(f'max_iter={self.max_iter}')
        logger.info('========================')

        split_path = self.input()['split-dataset'].path
        tasks_path = self.input()['tasks'].path
        
        # Retrieve val.csv and test.csv from the split-dataset folder, then the csv files from the tasks folder
        val_df = pd.read_csv(os.path.join(split_path, "val.csv"))
        test_df = pd.read_csv(os.path.join(split_path, "test.csv"))
        task_files = sorted(glob.glob(os.path.join(tasks_path, "task_*.csv")))

        label_map = {}
        next_class_index = 0
        gmms = {}
        metrics = []

        class_sample_counts = {}  # dict for prior computation
        first_task_mean = None
        first_task_std = None

        for i, task_file in enumerate(task_files):
            task_df = pd.read_csv(task_file)

            current_attacks = set(task_df["attack_type"].unique()) - {"benign", "Benign", "normal", "Normal"} 
            assert len(current_attacks) == 1, f"Task file {task_file} must contain exactly 1 attack type"
            current_attack = current_attacks.pop()

            # Assign index if first time seeing this attack type
            if current_attack not in label_map:
                label_map[current_attack] = next_class_index
                next_class_index += 1

            class_idx = label_map[current_attack]
            
            # --- TRAINING SET FROM THE TASK CONSIDERING JUST THE CURRENT ATTACK, NO BENIGN TRAFFIC ---
            task_df = task_df[task_df["attack_type"] == current_attack]
            X_train = task_df.drop(columns=["attack", "attack_type"])
            #y_train = np.full(len(X_train), class_idx)

            # --- VALIDATION SET WITH JUST THE CURRENT ATTACK ---
            val_task_df = val_df[val_df["attack_type"] == current_attack]
            X_val = val_task_df.drop(columns=["attack", "attack_type"])
            #y_val = np.full(len(X_val), class_idx)

            # --- CUMULATIVE TEST SET WITH THE SEEN ATTACKS UP TO NOW ---
            seen_attack_types = set(label_map.keys())
            test_seen_df = test_df[test_df["attack_type"].isin(seen_attack_types)]
            X_test = test_seen_df.drop(columns=["attack", "attack_type"])
            y_test = test_seen_df["attack_type"].map(label_map)

            # Identify columns to normalize: numeric columns, keep also constant ones since they may change among different tasks!
            columns_to_normalize = [numeric_column
                                    for numeric_column in list(X_train.select_dtypes(include=['float', 'int']).columns)]

            # Retrieve mean and std if it's the first task
            if first_task_mean is None:
                first_task_mean = X_train[columns_to_normalize].mean()
                first_task_std = X_train[columns_to_normalize].std()
            
            # --- Z-SCORE NORMALIZATION USING MEAN & STD FROM 1st TASK ---
            for col in columns_to_normalize:
                mean = first_task_mean[col]
                std = first_task_std[col]
                # Avoid NaN or something that isn't ok for training a GMM
                if pd.isna(mean) or mean < 1e-8 or pd.isna(std) or std < 1e-8:
                    X_train[col] = 0.0
                    X_val[col] = 0.0
                    X_test[col] = 0.0
                else:
                    X_train[col] = (X_train[col] - mean) / std
                    X_val[col] = (X_val[col] - mean) / std
                    X_test[col] = (X_test[col] - mean) / std

            # Train new GMM for current attack
            gmm = BayesianGaussianMixture(n_components=self.n_components,
                                          covariance_type=self.covariance_type,
                                          reg_covar=self.reg_covar,
                                          weight_concentration_prior_type=self.weight_concentration_prior_type,
                                          weight_concentration_prior=self.weight_concentration_prior,
                                          max_iter=self.max_iter,
                                          random_state=42)
            gmm.fit(X_train)
            gmms[class_idx] = gmm

            # ELBO
            elbo = gmm.lower_bound_

            # Update class sample count (manual check)
            if class_idx not in class_sample_counts:
                class_sample_counts[class_idx] = 0
            class_sample_counts[class_idx] += len(X_train)
            total_samples_seen = sum(class_sample_counts.values())

            # Compute log-likelihood + log-prior for each class
            log_likelihoods = np.zeros((X_test.shape[0], len(gmms)))
            for gmm_class_idx, gmm_model in gmms.items():
                log_likelihood = gmm_model.score_samples(X_test)
                prior_prob = class_sample_counts[gmm_class_idx] / total_samples_seen
                log_prior = np.log(prior_prob)
                log_likelihoods[:, gmm_class_idx] = log_likelihood + log_prior

            y_pred = np.argmax(log_likelihoods, axis=1)

            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            logger.info(f'Task {i + 1} -> Accuracy: {accuracy:.4f} | F1 (Macro): {f1_macro:.4f} | F1 (Weighted): {f1_weighted:.4f}')
            
            metrics.append({
                "dataset": self.dataset_name,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                "weight_concentration_prior_type": self.weight_concentration_prior_type,
                "weight_concentration_prior": self.weight_concentration_prior,
                "max_iter": self.max_iter,
                "task": i + 1,
                "attack": current_attack,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "elbo": elbo
            })

        # --- SAVE MODEL AND METRICS ---
        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, 'Continual_Learning_BayesianGMM.pkl')
        continual_metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['bayesian_continual_metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(continual_metrics_csv), exist_ok=True)

        # Save model as pickle
        with open(model_file, 'wb') as f:
            pickle.dump(gmms, f)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        logger.info('Metrics:')
        logger.info(f'\n{metrics_df}')

        with open(continual_metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {continual_metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class FullDatasetSVDGMM(luigi.Task):
    """
    SVD applied on data + Supervised Gaussian Mixture Model (GMM) for multi-class classification
    """

    dataset_name = luigi.Parameter()

    # Filter only attack data
    attack_only = luigi.BoolParameter(default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING) 

    # Percentage of training data to use
    train_percentage = luigi.IntParameter(default=100) 

    # Max number of components that can be used globally to represent attack types
    # If tune_n_components is false, the number is fixed, else it's the upper bound (tuning starting from 1)
    max_components = luigi.IntParameter(default=5)

    # Type of covariance matrix
    covariance_type = luigi.Parameter(default="full")

    # Regularization for covariance matrix
    reg_covar = luigi.FloatParameter(default=1e-2)

    # Decide whether to tune the number of components globally in GMMs or not, by default yes
    tune_n_components = luigi.BoolParameter(default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    # Metric (AIC, f1_score or accuracy) used for tuning on validation set if tune_n_components is true
    # If not tuning, it's calculated once since the number of components is fixed
    selection_metric = luigi.Parameter(default="AIC")

    # Number of components used in SVD
    n_components_SVD = luigi.IntParameter(default=30)

    def requires(self):
        # Train, validation and test are needed 
        class FakeTask(luigi.Task):
            def output(_):
                return DirectoryTarget(get_full_dataset_rel_path(self.dataset_name,
                                                                 cfg['split_closed_set_rel_folder']))

        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info('====== PARAMETERS FOR SVD Full-Dataset GMM ======')
        logger.info(f'attack_only={self.attack_only}')
        logger.info(f'train_percentage={self.train_percentage}')
        logger.info(f'max_components={self.max_components}')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'tune_n_components={self.tune_n_components}')
        logger.info(f'selection_metric={self.selection_metric}')
        logger.info(f'n_components_SVD={self.n_components_SVD}')
        logger.info('========================')

        # Load the train dataset
        train_df = pd.read_csv(os.path.join(self.input().path, 'train.csv'))

        logger.info(f'Loaded train set from {self.input().path}/train.csv')

        # Load the validation dataset
        val_df = pd.read_csv(os.path.join(self.input().path, 'val.csv'))

        logger.info(f'Loaded validation set from {self.input().path}/val.csv')

        # Load the test dataset
        test_df = pd.read_csv(os.path.join(self.input().path, 'test.csv'))

        logger.info(f'Loaded test set from {self.input().path}/test.csv')

        if self.attack_only:
            train_df = train_df[train_df['attack'] == True]
            val_df = val_df[val_df['attack'] == True]
            test_df = test_df[test_df['attack'] == True]
            logger.info(f'Filtered only attack data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')

        # Apply percentage-based sampling on training set
        if self.train_percentage > 0 and self.train_percentage < 100:
            train_df, _ = train_test_split(train_df, train_size=self.train_percentage / 100, stratify=train_df["attack"], random_state=42)
            logger.info(f'Using only {self.train_percentage}% of training data: {len(train_df)} samples')
        
        # Extract features
        X_train = train_df.drop(columns=["attack", "attack_type"])
        X_val = val_df.drop(columns=["attack", "attack_type"])
        X_test = test_df.drop(columns=["attack", "attack_type"])

        # Apply SVD
        svd = TruncatedSVD(n_components=self.n_components_SVD, random_state=42)
        X_train = pd.DataFrame(svd.fit_transform(X_train)).reset_index(drop=True)
        X_val = pd.DataFrame(svd.transform(X_val)).reset_index(drop=True)
        X_test = pd.DataFrame(svd.transform(X_test)).reset_index(drop=True)

        logger.info(f'Applied SVD with n_components = {self.n_components_SVD}')

        logger.info(f'Transformed training set preview:')
        logger.info(f'\n{X_train.head()}')

        # Extract labels
        y_train = train_df["attack_type"]
        y_val = val_df["attack_type"]
        y_test = test_df["attack_type"]

        # Encode attack types to numeric values
        attack_types = sorted(y_train.unique())
        attack_mapping = {attack: i for i, attack in enumerate(attack_types)}
        y_train = y_train.map(attack_mapping).reset_index(drop=True)
        y_val = y_val.map(attack_mapping).reset_index(drop=True)
        y_test = y_test.map(attack_mapping).reset_index(drop=True)

        logger.info(f'Mapped attack types to numerical labels: {attack_mapping}')

        ##### --- GLOBAL TUNING LOOP --- #####
        best_class_gmms = None       # Dictionary of class_idx -> best GMM
        best_score = -np.inf         # Best validation score observed so far
        best_n_components = None     # Number of components corresponding to best_score

        # If tuning try number of components from 1 to max_components, else just one iteration (max_components itself)
        for n_components in range(1, self.max_components + 1) if self.tune_n_components else [self.max_components]:
            
            logger.info(f'Trying n_components = {n_components}')

            class_gmms = {}

            # Train one GMM per class
            for attack_type in attack_types:
                class_idx = attack_mapping[attack_type]
                X_train_class = X_train[y_train == class_idx]

                gmm = GaussianMixture(n_components=n_components,
                                      covariance_type=self.covariance_type,
                                      reg_covar=self.reg_covar,
                                      random_state=42)
                gmm.fit(X_train_class)
                class_gmms[class_idx] = gmm

            # Use all GMMs to predict on validation set
            log_likelihoods = np.zeros((X_val.shape[0], len(class_gmms)))
            for class_idx, gmm in class_gmms.items():
                log_likelihoods[:, class_idx] = gmm.score_samples(X_val)
            y_val_pred = np.argmax(log_likelihoods, axis=1)

            # Compute validation score based on selection metric
            if self.selection_metric.lower() == "accuracy":
                score = accuracy_score(y_val, y_val_pred)
            elif self.selection_metric.lower() == "f1_score":
                score = f1_score(y_val, y_val_pred, average="macro")
            else: # == "aic":
                # We want to minimize AIC not maximize, so we use a negative sign
                score = -sum(class_gmms[class_idx].aic(X_val[y_val == class_idx])
                            for _, class_idx in attack_mapping.items())

            if self.selection_metric.lower() == "aic":
                logger.info(f'n_components={n_components} -> Validation AIC = {-score} (minimization)')
            else:
                logger.info(f'n_components={n_components} -> Validation {self.selection_metric} = {score} (maximization)')

            # Update best model if better than previous ones
            if score > best_score:
                best_score = score
                best_class_gmms = class_gmms
                best_n_components = n_components
                logger.info(f'Best n_components so far: {n_components}')

        ##### --- PREDICT ON TEST SET --- #####
        log_likelihoods = np.zeros((X_test.shape[0], len(best_class_gmms)))
        for class_idx, gmm in best_class_gmms.items():
            log_likelihoods[:, class_idx] = gmm.score_samples(X_test)
        y_test_pred = np.argmax(log_likelihoods, axis=1)

        # Compute final test metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average="macro")
        recall = recall_score(y_test, y_test_pred, average="macro")
        f1 = f1_score(y_test, y_test_pred, average="macro")
        f1_weighted = f1_score(y_test, y_test_pred, average="weighted")

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score (Macro): {f1}')
        logger.info(f'-> F1-Score (Weighted): {f1_weighted}')
        

        ##### --- SAVE MODEL AND METRICS --- #####
        model_name = f'Full-Dataset_SVD_SupervisedGMM{"_AttackOnly" if self.attack_only else ""}_{self.train_percentage}%' + \
                    f'_{"Tune" if self.tune_n_components else "Fixed"}-{best_n_components}_{self.selection_metric}'

        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, f'{model_name}.pkl')
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        # Save model as pickle
        with open(model_file, 'wb') as f:
            pickle.dump(best_class_gmms, f)

        # Save evaluation metrics to CSV
        metrics_df = pd.DataFrame([{
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f1_score_weighted": f1_weighted,
            "roc_auc": "N/A (Multi)",
            "best_hyperparameters": str({
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                "selection_metric": self.selection_metric,
                "tune_components": self.tune_n_components,
                "n_components": best_n_components,
                "n_components_SVD": self.n_components_SVD
            }),
        }])

        with open(metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')



class ContinualBNNPlusGMM(luigi.Task):
    """
    Final continual model: binary BNN (Laplace Redux) + GMM multi-class only on attack predictions.
    """

    dataset_name = luigi.Parameter()

    batch_size = luigi.IntParameter(default=128)

    learning_rate = luigi.FloatParameter(default=0.001)

    lam = luigi.FloatParameter(default=1.0)

    permute_tasks = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    covariance_type = luigi.Parameter(default="full")
    
    reg_covar = luigi.FloatParameter(default=1e-6)

    n_components = luigi.IntParameter(default=3)

    train_percentage = luigi.IntParameter(default=100)

    def requires(self):
        # The "split-dataset" (test.csv) and "tasks" folders are needed
        class FakeTask(luigi.Task):
            def output(_):
                return {'split-dataset': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                cfg['split_closed_set_rel_folder'])),
                        'tasks': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                           cfg['tasks_rel_folder']))}
        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f'====== PARAMETERS (DATASET {self.dataset_name}) ======')
        logger.info(f'batch_size={self.batch_size}')
        logger.info(f'learning_rate={self.learning_rate}')
        logger.info(f'lam={self.lam}')
        logger.info(f'permute_tasks={self.permute_tasks}')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'n_components={self.n_components}')
        logger.info(f'train_percentage={self.train_percentage}')
        logger.info('========================')

        # === Load data and task files ===
        split_path = self.input()['split-dataset'].path
        tasks_path = self.input()['tasks'].path
        test_df = pd.read_csv(os.path.join(split_path, "test.csv"))
        task_files = sorted(glob.glob(os.path.join(tasks_path, "task_*.csv")))
        if self.permute_tasks:
            random.shuffle(task_files)

        # === Initialize state ===
        seen_attacks = []
        class_idx_map = {}  # benign is handled via is_benign (class 0)
        next_class_idx = 1
        gmm_models = {}
        class_counts = {}
        metrics = []

        first_task_mean, first_task_std = None, None
        model, laplace = None, None

        def is_benign(attack_type):
            return str(attack_type).strip().lower() in {"benign", "normal"}

        def normalize(df):
            for col in columns_to_normalize:
                mean, std = first_task_mean[col], first_task_std[col]
                df[col] = 0.0 if pd.isna(mean) or pd.isna(std) or std < 1e-8 else (df[col] - mean) / std
            return df

        def predict_system(X_np, y_true_cls=None):  # y_true_cls is for debug
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(X_np, dtype=torch.float32)), dim=1).detach().numpy()

            p_attack = probs[:, 1]
            binary_preds = (p_attack > 0.5).astype(int)

            if y_true_cls is not None:
                true_attacks = (y_true_cls != 0)
                detected_attacks = binary_preds == 1
                TP = np.sum(true_attacks & detected_attacks)
                FN = np.sum(true_attacks & (binary_preds == 0))
                FP = np.sum((~true_attacks) & (binary_preds == 1))
                TN = np.sum((~true_attacks) & (binary_preds == 0))
                logger.info(f"[BNN] TP={TP}, FN={FN}, FP={FP}, TN={TN}")
                logger.info(f"[BNN] P(attack) stats: mean={p_attack.mean():.3f}, min={p_attack.min():.3f}, max={p_attack.max():.3f}")

            y_pred = np.full(len(X_np), 0)

            if np.any(binary_preds == 1):
                X_attack = X_np[binary_preds == 1]
                cls_keys = list(gmm_models.keys())
                log_likelihoods = np.zeros((X_attack.shape[0], len(cls_keys)))
                total = sum(class_counts.values())

                for idx, cls in enumerate(cls_keys):
                    gmm = gmm_models[cls]
                    log_likelihoods[:, idx] = gmm.score_samples(X_attack) + np.log(class_counts[cls] / total)

                gmm_preds = np.argmax(log_likelihoods, axis=1)
                y_attack = np.array([cls_keys[i] for i in gmm_preds])
                y_pred[binary_preds == 1] = y_attack

            return y_pred


        # === Iterate over tasks ===
        for i, task_file in enumerate(task_files):
            logger.info(f"=========== TASK {i+1} ===========")
            task_df = pd.read_csv(task_file)

            current_attacks = set(a for a in task_df["attack_type"].unique() if not is_benign(a))
            assert len(current_attacks) == 1
            current_attack = current_attacks.pop()
            seen_attacks.append(current_attack)
            logger.info(f"Current attack: {current_attack}")

            if current_attack not in class_idx_map:
                class_idx_map[current_attack] = next_class_idx
                next_class_idx += 1

            # === Prepare training data ===
            if 0 < self.train_percentage < 100:
                task_df, _ = train_test_split(
                    task_df,
                    train_size=self.train_percentage / 100,
                    stratify=task_df["attack"],
                    random_state=42
                )
                logger.info(f'Using only {self.train_percentage}% of training data for task {i+1}: {len(task_df)} samples')
            X_train_df = task_df.drop(columns=["attack", "attack_type"])
            y_train_bin = task_df["attack"].astype(int).values
            y_train_cls = np.full(len(X_train_df), class_idx_map[current_attack])

            columns_to_normalize = list(X_train_df.select_dtypes(include=["float", "int"]).columns)
            if first_task_mean is None:
                first_task_mean = X_train_df[columns_to_normalize].mean()
                first_task_std = X_train_df[columns_to_normalize].std()
            X_train_df = normalize(X_train_df)

            X_train = torch.tensor(X_train_df.values.astype(np.float32))
            y_train_bin_t = torch.tensor(y_train_bin, dtype=torch.long)
            train_loader = DataLoader(TensorDataset(X_train, y_train_bin_t), batch_size=self.batch_size, shuffle=True)

            # === Initialize and train BNN ===
            if model is None:
                model = MLP(X_train.shape[1])
                logger.info(f"Initialized MLP with input dim = {X_train.shape[1]}")

            logger.info("Training BNN...")
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(10):
                model.train()
                for xb, yb in train_loader:
                    opt.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    if laplace is not None:
                        theta = parameters_to_vector(model.parameters())
                        log_prior = laplace.log_prob(theta) / len(xb)
                        loss = loss - self.lam * log_prior
                    loss.backward()
                    opt.step()

            # === Fit Laplace ===
            logger.info("Fitting Laplace...")
            laplace = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='kron', backend=AsdlGGN)
            laplace.fit(train_loader)
            laplace.prior_mean = laplace.mean.clone()
            laplace.optimize_prior_precision()
            logger.info("Laplace fit complete.")

            # === Train GMMs ===
            X_train_attack = X_train_df.values[y_train_bin == 1]
            y_train_attack = y_train_cls[y_train_bin == 1]
            logger.info(f"{len(X_train_attack)} training samples labeled as attack (for GMM training)")

            for cls in np.unique(y_train_attack):
                X_cls = X_train_attack[y_train_attack == cls]
                gmm = GaussianMixture(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    reg_covar=self.reg_covar,
                                    random_state=42)
                gmm.fit(X_cls)
                gmm_models[cls] = gmm
                class_counts[cls] = class_counts.get(cls, 0) + len(X_cls)
                logger.info(f"Trained GMM for class {cls} with {len(X_cls)} samples")

            # === Evaluate cumulative ===
            test_seen_df = test_df[test_df["attack_type"].apply(lambda x: x in seen_attacks or is_benign(x))].copy()
            X_test_np = normalize(test_seen_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
            y_true_cls = test_seen_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
            y_pred_cls = predict_system(X_test_np, y_true_cls)

            scores = {
                "accuracy": accuracy_score(y_true_cls, y_pred_cls),
                "f1_macro": f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
            }

            # === Evaluate current ===
            current_df = test_df[test_df["attack_type"].apply(lambda x: x == current_attack or is_benign(x))].copy()
            X_curr_np = normalize(current_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
            y_curr = current_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
            y_pred_curr = predict_system(X_curr_np, y_curr)
            scores_current = {
                "accuracy": accuracy_score(y_curr, y_pred_curr),
                "f1_macro": f1_score(y_curr, y_pred_curr, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_curr, y_pred_curr, average="weighted", zero_division=0)
            }

            # === Evaluate previous ===
            if i > 0:
                previous_df = test_df[test_df["attack_type"].apply(lambda x: x in seen_attacks[:-1] or is_benign(x))].copy()
                X_prev_np = normalize(previous_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
                y_prev = previous_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
                y_pred_prev = predict_system(X_prev_np, y_prev)
                scores_previous = {
                    "accuracy": accuracy_score(y_prev, y_pred_prev),
                    "f1_macro": f1_score(y_prev, y_pred_prev, average="macro", zero_division=0),
                    "f1_weighted": f1_score(y_prev, y_pred_prev, average="weighted", zero_division=0)
                }
            else:
                scores_previous = {k: None for k in ["accuracy", "f1_macro", "f1_weighted"]}

            # === Save metrics for task ===
            metrics.append({
                "dataset": f"{self.dataset_name}_{self.train_percentage}%",
                "task": i + 1,
                "attack": current_attack,
                "lambda": self.lam,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "permute_tasks": self.permute_tasks,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                **{f"cumulative_{k}": scores[k] for k in scores},
                **{f"current_{k}": scores_current[k] for k in scores_current},
                **{f"previous_{k}": scores_previous[k] for k in scores_previous}
            })

        # === Save all results ===
        csv_path = get_full_model_rel_path(self.dataset_name, cfg["bnn_gmm_continual_metrics_rel_filename"])
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(metrics)
        logger.info(f"\n{df}")
        with open(csv_path, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')
        logger.info(f"Saved metrics to {csv_path}")
        logger.info(f"Finished task {self.__class__.__name__}")


class FullDatasetNNMulticlass(luigi.Task):
    """
    Full-dataset multi-class classification using a NN (including benign)
    """

    dataset_name = luigi.Parameter()
    batch_size = luigi.IntParameter(default=128)
    learning_rate = luigi.FloatParameter(default=0.001)

    def requires(self):
        class FakeTask(luigi.Task):
            def output(_):
                return DirectoryTarget(get_full_dataset_rel_path(self.dataset_name,
                                                                 cfg['split_closed_set_rel_folder']))
        return FakeTask()

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')
        logger.info(f'batch_size={self.batch_size}, learning_rate={self.learning_rate}')

        torch.manual_seed(np.random.default_rng().integers(0, 2**32 - 1))

        # Load datasets
        path = self.input().path
        train_df = pd.read_csv(os.path.join(path, 'train.csv'))
        test_df = pd.read_csv(os.path.join(path, 'test.csv'))

        # Features and labels
        X_train = train_df.drop(columns=["attack", "attack_type"])
        X_test = test_df.drop(columns=["attack", "attack_type"])

        y_train = train_df["attack_type"]
        y_test = test_df["attack_type"]

        # Encode class labels
        class_labels = sorted(y_train.unique())
        class_map = {label: idx for idx, label in enumerate(class_labels)}
        y_train = y_train.map(class_map).values
        y_test = y_test.map(class_map).values
        num_classes = len(class_map)

        logger.info(f'Mapped attack types to class indices: {class_map}')

        # Normalization
        numeric_cols = list(X_train.select_dtypes(include=['float', 'int']).columns)
        mean, std = X_train[numeric_cols].mean(), X_train[numeric_cols].std()
        for df in [X_train, X_test]:
            df[numeric_cols] = (df[numeric_cols] - mean) / std

        # Torch tensors
        X_train_t = torch.tensor(X_train.values.astype(np.float32))
        X_test_t = torch.tensor(X_test.values.astype(np.float32))
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)

        # Model
        model = MLP(X_train_t.shape[1], output_dim=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # Training
        for epoch in range(10):
            model.train()
            correct, total, total_loss = 0, 0, 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                correct += (out.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
            logger.info(f'Epoch {epoch+1} | Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total:.4f}')

        # Evaluation
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_test_t), dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1 (Macro): {f1}')
        logger.info(f'-> F1 (Weighted): {f1_weighted}')

        # Save model and metrics
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['nn_multi_metrics_rel_filename'])
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        metrics_df = pd.DataFrame([{
            "dataset": self.dataset_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1,
            "f1_weighted": f1_weighted,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }])
        with open(metrics_csv, 'a') as f:
            metrics_df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')
        logger.info(f'Finished task {self.__class__.__name__}')


class ContinualBNNPlusGMMFullNorm(luigi.Task):
    """
    Final continual model: binary BNN (Laplace Redux) + GMM multi-class only on attack predictions.

    The normalization is performed using the whole training set, for comparison purposes.
    """

    dataset_name = luigi.Parameter()

    batch_size = luigi.IntParameter(default=128)

    learning_rate = luigi.FloatParameter(default=0.001)

    lam = luigi.FloatParameter(default=1.0)

    permute_tasks = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    covariance_type = luigi.Parameter(default="full")
    
    reg_covar = luigi.FloatParameter(default=1e-6)

    n_components = luigi.IntParameter(default=3)

    def requires(self):
        # The "split-dataset" (train.csv and test.csv) and "tasks" folders are needed
        class FakeTask(luigi.Task):
            def output(_):
                return {'split-dataset': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                cfg['split_closed_set_rel_folder'])),
                        'tasks': DirectoryTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                           cfg['tasks_rel_folder']))}
        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f'====== PARAMETERS (DATASET {self.dataset_name}) ======')
        logger.info(f'batch_size={self.batch_size}')
        logger.info(f'learning_rate={self.learning_rate}')
        logger.info(f'lam={self.lam}')
        logger.info(f'permute_tasks={self.permute_tasks}')
        logger.info(f'covariance_type={self.covariance_type}')
        logger.info(f'reg_covar={self.reg_covar}')
        logger.info(f'n_components={self.n_components}')
        logger.info('========================')

        # === Load data and task files ===
        split_path = self.input()['split-dataset'].path
        tasks_path = self.input()['tasks'].path
        train_df = pd.read_csv(os.path.join(split_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(split_path, "test.csv"))
        task_files = sorted(glob.glob(os.path.join(tasks_path, "task_*.csv")))
        if self.permute_tasks:
            random.shuffle(task_files)

        # === Initialize state ===
        seen_attacks = []
        class_idx_map = {}  # benign is handled via is_benign (class 0)
        next_class_idx = 1
        gmm_models = {}
        class_counts = {}
        metrics = []

        # === Normalization here is performed using the whole train ===
        columns_to_normalize = list(train_df.drop(columns=["attack", "attack_type"]).select_dtypes(include=["float", "int"]).columns)
        means = train_df[columns_to_normalize].mean()
        stds = train_df[columns_to_normalize].std()

        model, laplace = None, None

        def is_benign(attack_type):
            return str(attack_type).strip().lower() in {"benign", "normal"}

        def normalize(df):
            for col in columns_to_normalize:
                mean, std = means[col], stds[col]
                df[col] = 0.0 if pd.isna(mean) or pd.isna(std) or std < 1e-8 else (df[col] - mean) / std
            return df

        def predict_system(X_np, y_true_cls=None):  # y_true_cls is for debug
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(X_np, dtype=torch.float32)), dim=1).detach().numpy()

            p_attack = probs[:, 1]
            binary_preds = (p_attack > 0.5).astype(int)

            if y_true_cls is not None:
                true_attacks = (y_true_cls != 0)
                detected_attacks = binary_preds == 1
                TP = np.sum(true_attacks & detected_attacks)
                FN = np.sum(true_attacks & (binary_preds == 0))
                FP = np.sum((~true_attacks) & (binary_preds == 1))
                TN = np.sum((~true_attacks) & (binary_preds == 0))
                logger.info(f"[BNN] TP={TP}, FN={FN}, FP={FP}, TN={TN}")
                logger.info(f"[BNN] P(attack) stats: mean={p_attack.mean():.3f}, min={p_attack.min():.3f}, max={p_attack.max():.3f}")

            y_pred = np.full(len(X_np), 0)

            if np.any(binary_preds == 1):
                X_attack = X_np[binary_preds == 1]
                cls_keys = list(gmm_models.keys())
                log_likelihoods = np.zeros((X_attack.shape[0], len(cls_keys)))
                total = sum(class_counts.values())

                for idx, cls in enumerate(cls_keys):
                    gmm = gmm_models[cls]
                    log_likelihoods[:, idx] = gmm.score_samples(X_attack) + np.log(class_counts[cls] / total)

                gmm_preds = np.argmax(log_likelihoods, axis=1)
                y_attack = np.array([cls_keys[i] for i in gmm_preds])
                y_pred[binary_preds == 1] = y_attack

            return y_pred


        # === Iterate over tasks ===
        for i, task_file in enumerate(task_files):
            logger.info(f"=========== TASK {i+1} ===========")
            task_df = pd.read_csv(task_file)

            current_attacks = set(a for a in task_df["attack_type"].unique() if not is_benign(a))
            assert len(current_attacks) == 1
            current_attack = current_attacks.pop()
            seen_attacks.append(current_attack)
            logger.info(f"Current attack: {current_attack}")

            if current_attack not in class_idx_map:
                class_idx_map[current_attack] = next_class_idx
                next_class_idx += 1

            # === Prepare training data ===
            X_train_df = task_df.drop(columns=["attack", "attack_type"])
            y_train_bin = task_df["attack"].astype(int).values
            y_train_cls = np.full(len(X_train_df), class_idx_map[current_attack])

            X_train_df = normalize(X_train_df)

            X_train = torch.tensor(X_train_df.values.astype(np.float32))
            y_train_bin_t = torch.tensor(y_train_bin, dtype=torch.long)
            train_loader = DataLoader(TensorDataset(X_train, y_train_bin_t), batch_size=self.batch_size, shuffle=True)

            # === Initialize and train BNN ===
            if model is None:
                model = MLP(X_train.shape[1])
                logger.info(f"Initialized MLP with input dim = {X_train.shape[1]}")

            logger.info("Training BNN...")
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(10):
                model.train()
                for xb, yb in train_loader:
                    opt.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    if laplace is not None:
                        theta = parameters_to_vector(model.parameters())
                        log_prior = laplace.log_prob(theta) / len(xb)
                        loss = loss - self.lam * log_prior
                    loss.backward()
                    opt.step()

            # === Fit Laplace ===
            logger.info("Fitting Laplace...")
            laplace = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='kron', backend=AsdlGGN)
            laplace.fit(train_loader)
            laplace.prior_mean = laplace.mean.clone()
            laplace.optimize_prior_precision()
            logger.info("Laplace fit complete.")

            # === Train GMMs ===
            X_train_attack = X_train_df.values[y_train_bin == 1]
            y_train_attack = y_train_cls[y_train_bin == 1]
            logger.info(f"{len(X_train_attack)} training samples labeled as attack (for GMM training)")

            for cls in np.unique(y_train_attack):
                X_cls = X_train_attack[y_train_attack == cls]
                gmm = GaussianMixture(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    reg_covar=self.reg_covar,
                                    random_state=42)
                gmm.fit(X_cls)
                gmm_models[cls] = gmm
                class_counts[cls] = class_counts.get(cls, 0) + len(X_cls)
                logger.info(f"Trained GMM for class {cls} with {len(X_cls)} samples")

            # === Evaluate cumulative ===
            test_seen_df = test_df[test_df["attack_type"].apply(lambda x: x in seen_attacks or is_benign(x))].copy()
            X_test_np = normalize(test_seen_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
            y_true_cls = test_seen_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
            y_pred_cls = predict_system(X_test_np, y_true_cls)

            scores = {
                "accuracy": accuracy_score(y_true_cls, y_pred_cls),
                "f1_macro": f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
            }

            # === Evaluate current ===
            current_df = test_df[test_df["attack_type"].apply(lambda x: x == current_attack or is_benign(x))].copy()
            X_curr_np = normalize(current_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
            y_curr = current_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
            y_pred_curr = predict_system(X_curr_np, y_curr)
            scores_current = {
                "accuracy": accuracy_score(y_curr, y_pred_curr),
                "f1_macro": f1_score(y_curr, y_pred_curr, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_curr, y_pred_curr, average="weighted", zero_division=0)
            }

            # === Evaluate previous ===
            if i > 0:
                previous_df = test_df[test_df["attack_type"].apply(lambda x: x in seen_attacks[:-1] or is_benign(x))].copy()
                X_prev_np = normalize(previous_df.drop(columns=["attack", "attack_type"])).values.astype(np.float32)
                y_prev = previous_df["attack_type"].apply(lambda x: class_idx_map.get(x) if not is_benign(x) else 0).values
                y_pred_prev = predict_system(X_prev_np, y_prev)
                scores_previous = {
                    "accuracy": accuracy_score(y_prev, y_pred_prev),
                    "f1_macro": f1_score(y_prev, y_pred_prev, average="macro", zero_division=0),
                    "f1_weighted": f1_score(y_prev, y_pred_prev, average="weighted", zero_division=0)
                }
            else:
                scores_previous = {k: None for k in ["accuracy", "f1_macro", "f1_weighted"]}

            # === Save metrics for task ===
            metrics.append({
                "dataset": self.dataset_name,
                "task": i + 1,
                "attack": current_attack,
                "lambda": self.lam,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "permute_tasks": self.permute_tasks,
                "n_components": self.n_components,
                "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar,
                **{f"cumulative_{k}": scores[k] for k in scores},
                **{f"current_{k}": scores_current[k] for k in scores_current},
                **{f"previous_{k}": scores_previous[k] for k in scores_previous}
            })

        # === Save all results ===
        csv_path = get_full_model_rel_path(self.dataset_name, cfg["full_norm_bnn_gmm_continual_metrics_rel_filename"])
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(metrics)
        logger.info(f"\n{df}")
        with open(csv_path, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False, lineterminator='\n')
        logger.info(f"Saved metrics to {csv_path}")
        logger.info(f"Finished task {self.__class__.__name__}")