import luigi
import logging
import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from sklearn.model_selection import train_test_split


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
}

# Retrieve a relative path w.r.t the dataset name in the datasets folder
# suffix is the substring of the path after the dataset folder
def get_full_dataset_rel_path(dataset_name, suffix):
    return f'{cfg["datasets_folder"]}/{dataset_name}/{suffix}'

# Retrieve a relative path w.r.t the dataset name in the models folder
# suffix is the substring of the path after the model folder
def get_full_model_rel_path(dataset_name, suffix):
    return f'{cfg["models_folder"]}/{dataset_name}/{suffix}'


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
        else:
            auc_roc = None # No AUROC for multi-classification

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
    attack_only = luigi.BoolParameter(default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING) 

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

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score: {f1}')

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