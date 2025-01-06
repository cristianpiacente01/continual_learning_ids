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


class FullDatasetRF(luigi.Task):
    """
    Random Forest binary classification on the full dataset
    """

    dataset_name = luigi.Parameter()
    tuning_min_samples_split = luigi.ListParameter(default=(2, 5, 10)) # Used for tuning
    tuning_min_samples_leaf = luigi.ListParameter(default=(1, 2, 4)) # Used for tuning
    tuning_iterations = luigi.IntParameter(default=10) # Used for tuning

    def requires(self):
        # preprocessed.csv is needed
        class FakeTask(luigi.Task):
            def output(_):
                return luigi.LocalTarget(get_full_dataset_rel_path(self.dataset_name, 
                                                                   cfg['preprocessed_rel_filename']))

        return FakeTask()


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read preprocessed.csv
        df = pd.read_csv(self.input().path)

        logger.info(f'Retrieved the preprocessed full dataset')

        # Extract features and labels
        X = df.drop(columns=["attack", "attack_type"]) # Features
        y = df["attack"] # Binary labels for binary classification

        ##### --- FROM FULL DATASET TO NORMALIZED TRAIN, VAL, TEST --- #####

        # Split into train, val and test (60-20-20)
        X_train, X_remainder, y_train, y_remainder = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder, test_size=0.5, random_state=42, stratify=y_remainder)

        logger.info(f'Split task into train ({len(X_train)}), val ({len(X_val)}), and test ({len(X_test)})')

        # Identify columns to normalize
        float_columns = X_train.select_dtypes(include=['float']).columns
        high_cardinality_int_columns = [col for col in X_train.select_dtypes(include=['int']).columns if X_train[col].nunique() > 10]
        columns_to_normalize = list(float_columns) + high_cardinality_int_columns

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
        y_test_proba = best_model.predict_proba(X_test)[:, 1] # For AUROC calculation

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auroc = roc_auc_score(y_test, y_test_proba)

        logger.info(f'Evaluation metrics:')
        logger.info(f'-> Accuracy: {accuracy}')
        logger.info(f'-> Precision: {precision}')
        logger.info(f'-> Recall: {recall}')
        logger.info(f'-> F1-Score: {f1}')
        logger.info(f'-> AUROC: {auroc}')

        ##### --- SAVE THE MODEL AND METRICS --- #####

        model_folder = get_full_model_rel_path(self.dataset_name, '')
        model_file = os.path.join(model_folder, 'full_dataset_RF_model.pkl') # model name is here
        metrics_csv = get_full_model_rel_path(self.dataset_name, cfg['metrics_rel_filename'])

        # Create the folders if they don't exist
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)

        # Save .pkl file
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)

        logger.info(f'Saved the model to {model_file}')

        # Create a DataFrame for the metrics
        metrics_df = pd.DataFrame([{
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auroc": auroc,
            "best_hyperparameters": str(best_params),
        }])

        # Append the data to metrics.csv
        with open(metrics_csv, 'a') as f: 
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info(f'Saved the metrics to {metrics_csv}')

        logger.info(f'Finished task {self.__class__.__name__}')


        