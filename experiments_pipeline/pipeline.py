import luigi
import logging
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle


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
    tuning_iterations = luigi.IntParameter(default=10) # Used for tuning

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
        columns_to_normalize = X_train.select_dtypes(include=['float', 'int']).columns

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
        rf = RandomForestClassifier(random_state=42)
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
        #TODO if binary AUC-ROC

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
            "best_hyperparameters": str(best_params),
        }])

        # Append the data to metrics.csv
        with open(metrics_csv, 'a') as f: 
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')

        logger.info(f'Finished task {self.__class__.__name__}')


