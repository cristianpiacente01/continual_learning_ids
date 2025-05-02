import luigi
import logging
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('data-prep-pipeline')

# Get configuration file
luigi.configuration.LuigiConfigParser.add_config_path("../luigi.cfg")
config = luigi.configuration.get_config()

# Dict that contains the parameters configurated in luigi.cfg
paths = {
    'datasets_folder': config.get('DataPreparationPipeline', 'datasets_folder'),
    'merged_rel_filename': config.get('DataPreparationPipeline', 'merged_rel_filename'),
    'preprocessed_rel_filename': config.get('DataPreparationPipeline', 'preprocessed_rel_filename'),
    'closed_set_rel_filename': config.get('DataPreparationPipeline', 'closed_set_rel_filename'),
    'ood_rel_filename': config.get('DataPreparationPipeline', 'ood_rel_filename'),
    'split_closed_set_rel_folder': config.get('DataPreparationPipeline', 'split_closed_set_rel_folder'),
    'tasks_rel_folder': config.get('DataPreparationPipeline', 'tasks_rel_folder'), 
}

# Specific dataset folder
dataset_folder = ''

# Init global variable if not already done
def init_global_var(dataset_name):
    global dataset_folder
    if dataset_folder == '':
        dataset_folder = f'{paths["datasets_folder"]}/{dataset_name}'

# Retrieve a relative path w.r.t the dataset name
# suffix is the substring of the path after the dataset folder
def get_full_rel_path(dataset_name, suffix):
    init_global_var(dataset_name) # Initialize the global variable dataset_folder
    return f'{dataset_folder}/{suffix}'


class CustomTarget(luigi.Target):
    """
    Luigi Target defined for checking if there's at least 1 file with the given extension in the given path
    """
    
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension

    # Luigi's complete method override
    def complete(self):
        return len(glob.glob(f"{self.path}/*{self.extension}")) > 0

    # Luigi's complete method override
    def exists(self):
        return len(glob.glob(f"{self.path}/*{self.extension}")) > 0
    

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



class MergeFiles(luigi.Task):
    """
    Create a merged csv using all the files for a specific dataset, given the dataset name and files extension
    """

    dataset_name = luigi.Parameter() # Mandatory (e.g. "CIC-IDS2017")
    input_file_extension = luigi.Parameter() # Mandatory (e.g. ".parquet")

    def requires(self):
        # At least 1 file with the given extension is needed
        class FakeTask(luigi.Task):
            def output(_):
                return CustomTarget(path=get_full_rel_path(self.dataset_name, ''),
                                    extension=self.input_file_extension)

        return FakeTask()
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        matching_files = glob.glob(f"{get_full_rel_path(self.dataset_name, '')}/*{self.input_file_extension}")

        # Merge the files
        if self.input_file_extension.lower().endswith("parquet"):
            data_frames = [pd.read_parquet(file) for file in matching_files]
        else: # .csv
            data_frames = [pd.read_csv(file) for file in matching_files]

        logger.info(f'Retrieved {len(matching_files)} files')
        
        merged_df = pd.concat(data_frames, ignore_index=True)

        # Create the output path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to .csv
        merged_df.to_csv(self.output().path, index=False)

        logger.info(f'Merged {len(matching_files)} files into the csv file {self.output().path}')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.dataset_name, 
                                                   paths['merged_rel_filename']))



class Preprocessing(luigi.Task):
    """
    Perform all the preprocessing steps (see README.md at the root level in the repository), given the features to drop and attack types to drop
    """

    dataset_name = luigi.Parameter()
    input_file_extension = luigi.Parameter()
    features_to_drop = luigi.ListParameter(default=())
    attack_types_to_drop = luigi.ListParameter(default=())

    def requires(self):
        # merged.csv is needed
        return MergeFiles(dataset_name=self.dataset_name,
                          input_file_extension=self.input_file_extension)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read merged.csv
        df = pd.read_csv(self.input().path)

        logger.info(f'Retrieved the merged dataset')

        ##### --- AUTOMATIC CREATION OF COLUMNS ATTACK_TYPE, ATTACK --- #####

        # Search the correct column for attack_type
        columns_to_check = ['attack_cat', 'attack', 'Label', 'label', 'type']
        selected_column = None
        for col in reversed(columns_to_check):
            if col in df.columns and df[col].dtype in ['object', 'category']:
                selected_column = col
                break

        # Rename to attack_type
        df.rename(columns={selected_column: 'attack_type'}, inplace=True)

        logger.info(f'Created the column attack_type')

        # Create the attack column by checking attack_type
        benign_values = ['benign', 'Benign', 'normal', 'Normal']
        df['attack'] = ~df['attack_type'].isin(benign_values)

        logger.info(f'Created the column attack')

        ##### --- DROP FEATURES --- #####

        # ListParameter creates a tuple instead of a list, convert it before using
        self.features_to_drop = list(self.features_to_drop)

        # Drop the features
        df.drop(columns=self.features_to_drop, inplace=True)

        logger.info(f'Dropped the features {self.features_to_drop}')

        ##### --- DROP CONSTANT FEATURES --- #####

        # Find the constant features
        constant_features = [col for col in df.columns if df[col].nunique() == 1]

        # Drop them
        df.drop(columns=constant_features, inplace=True)

        logger.info(f'Dropped the constant features {constant_features}')

        ##### --- DROP SPECIFIC ATTACK TYPES --- #####

        # ListParameter creates a tuple instead of a list, convert it before using
        self.attack_types_to_drop = list(self.attack_types_to_drop)

        # Drop rows with attack_type in the given list parameter
        df = df.loc[~df['attack_type'].isin(self.attack_types_to_drop)]

        logger.info(f'Dropped the attack types {self.attack_types_to_drop}')

        ##### --- DATA CLEANING --- #####

        # Count the number of duplicate rows
        num_duplicate = df.duplicated().sum()

        # Drop duplicate rows
        df = df[~df.duplicated()]

        logger.info(f'Dropped {num_duplicate} duplicate rows')

        # Fix negative and infinite values
        df = df.replace([np.inf, -np.inf], np.nan) # Replace inf/-inf with NaN temporarily
        for col in df.columns:
            # Numeric columns only
            if df[col].dtype not in ['object', 'category', 'bool']:
                # Get the max value > 0 and use it to replace NaN
                max_val = df[df[col] > 0][col].max()
                df[col] = df[col].fillna(max_val)  
                # Get the min value >= 0 and use it to replace negative values
                min_val = df[df[col] >= 0][col].min()
                df[col] = df[col].apply(lambda x: min_val if x < 0 else x)

        logger.info(f'Resolved missing, inf and negative values')

        ##### --- CASTING INT TYPES PROPERLY --- #####

        # Cast eligible float columns to int
        for col in df.select_dtypes(include=['float']).columns:
            # Check if all values in the column are integers
            if np.all(df[col] == df[col].astype(int)):
                df[col] = df[col].astype(int)
                logger.info(f'Casted the column {col} to int')

        ##### --- ENCODING FOR CATEGORICAL FEATURES --- #####

        # List of columns to check for one-hot encoding
        target_columns = ['prot', 'Protocol', 'ip_prot', 'protocol_type', 
                        'service', 'flag', 'proto', 'state']
        
        # Find columns to encode in the list or of type 'object' or 'category' (excluding the target)
        columns_to_encode = [col for col in df.columns 
                            if (col in target_columns or df[col].dtype in ['object', 'category'])
                            and col != 'attack_type']
        
        # Perform one-hot encoding
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

        logger.info(f'One-hot encoded the features {columns_to_encode}')
        
        ##### --- CASTING BOOL TYPES PROPERLY --- #####
        
        # Cast integer columns with min value 0 and max value 1 to bool
        columns_to_cast = [col for col in df.columns
                        if pd.api.types.is_integer_dtype(df[col]) and df[col].min() == 0 and df[col].max() == 1]
        df[columns_to_cast] = df[columns_to_cast].astype(bool)

        logger.info(f'Casted the columns {columns_to_cast} to bool')

        # Create the output path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to .csv
        df.to_csv(self.output().path, index=False)

        logger.info(f'Preprocessing was successful, the output file is {self.output().path}')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.dataset_name, 
                                                   paths['preprocessed_rel_filename']))



class SplitDistributions(luigi.Task):
    """
    Split the dataset into Closed-Set and Out-of-Distribution distributions based on a threshold
    """

    dataset_name = luigi.Parameter()
    input_file_extension = luigi.Parameter()
    features_to_drop = luigi.ListParameter(default=())
    attack_types_to_drop = luigi.ListParameter(default=())
    threshold = luigi.IntParameter(default=3000) # Threshold for splitting

    def requires(self):
        # preprocessed.csv is needed
        return Preprocessing(dataset_name=self.dataset_name,
                             input_file_extension=self.input_file_extension,
                             features_to_drop=self.features_to_drop,
                             attack_types_to_drop=self.attack_types_to_drop)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Load the preprocessed dataset
        df = pd.read_csv(self.input().path)

        logger.info(f'Loaded the preprocessed dataset from {self.input().path}')

        # Count the number of samples per class in 'attack_type'
        class_counts = df['attack_type'].value_counts()

        logger.info(f'Class counts: \n{class_counts}')

        # Separate classes based on the threshold
        top_classes = class_counts[class_counts >= self.threshold].index.tolist()
        unknown_classes = class_counts[class_counts < self.threshold].index.tolist()

        logger.info(f'Top classes (Closed-Set): {top_classes}')
        logger.info(f'Unknown classes (Out-of-Distribution): {unknown_classes}')

        # Create Closed-Set distribution
        closed_set_df = df[df['attack_type'].isin(top_classes)]

        # Create Out-of-Distribution
        ood_df = df[~df['attack_type'].isin(top_classes)]
        ood_df['attack_type'] = 'unknown'

        # Create the output path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output()['closed_set'].path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output()['ood'].path), exist_ok=True)
        
        # Save to .csv
        closed_set_df.to_csv(self.output()['closed_set'].path, index=False)
        ood_df.to_csv(self.output()['ood'].path, index=False)

        logger.info(f'Saved Closed-Set distribution to {self.output()["closed_set"].path}')

        logger.info(f'Saved Out-of-Distribution to {self.output()["ood"].path}')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return {
            'closed_set': luigi.LocalTarget(get_full_rel_path(self.dataset_name, paths['closed_set_rel_filename'])),
            'ood': luigi.LocalTarget(get_full_rel_path(self.dataset_name, paths['ood_rel_filename']))
        }



class SplitClosedSet(luigi.Task):
    """
    Split the Closed-Set dataset into train, val, and test sets (60-20-20%).
    """
    
    dataset_name = luigi.Parameter()
    input_file_extension = luigi.Parameter()
    features_to_drop = luigi.ListParameter(default=())
    attack_types_to_drop = luigi.ListParameter(default=())
    threshold = luigi.IntParameter(default=3000)

    def requires(self):
        # The Closed-Set is needed
        return SplitDistributions(dataset_name=self.dataset_name,
                                  input_file_extension=self.input_file_extension,
                                  features_to_drop=self.features_to_drop,
                                  attack_types_to_drop=self.attack_types_to_drop,
                                  threshold=self.threshold)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Load the Closed-Set dataset
        df = pd.read_csv(self.input()['closed_set'].path)

        logger.info(f'Loaded Closed-Set dataset from {self.input()["closed_set"].path}')

        # Split into train (60%), val (20%), and test (20%)
        train_df, remainder_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['attack'])
        val_df, test_df = train_test_split(remainder_df, test_size=0.5, random_state=42, stratify=remainder_df['attack'])

        logger.info(f'Split dataset into train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})')

        # Save splits
        os.makedirs(self.output().path, exist_ok=True)
        train_df.to_csv(os.path.join(self.output().path, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.output().path, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.output().path, 'test.csv'), index=False)

        logger.info(f'Dataset split into train-val-test successfully in directory {self.output().path}')


    def output(self):
        return DirectoryTarget(get_full_rel_path(self.dataset_name, paths['split_closed_set_rel_folder']))



class CreateTasks(luigi.Task):
    """
    Create tasks from the train set with different sampling strategies.
    """

    dataset_name = luigi.Parameter()
    input_file_extension = luigi.Parameter()
    features_to_drop = luigi.ListParameter(default=())
    attack_types_to_drop = luigi.ListParameter(default=())
    threshold = luigi.IntParameter(default=3000)
    sampling_strategy = luigi.Parameter(default='natural') # 'natural', 'balanced_attacks', 'fully_balanced'

    def requires(self):
        # Train, validation and test are needed
        return SplitClosedSet(dataset_name=self.dataset_name,
                              input_file_extension=self.input_file_extension,
                              features_to_drop=self.features_to_drop,
                              attack_types_to_drop=self.attack_types_to_drop,
                              threshold=self.threshold)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Load the train dataset
        train_df = pd.read_csv(os.path.join(self.input().path, 'train.csv'))

        logger.info(f'Loaded train set from {self.input().path}/train.csv')

        # Columns list to normalize later
        #columns_to_normalize = list(train_df.select_dtypes(include=['float', 'int']).columns)

        # Separate benign and attack samples
        benign_df = train_df[train_df['attack'] == False]
        attack_dfs = {attack_type: train_df[train_df['attack_type'] == attack_type]
                      for attack_type in train_df['attack_type'].unique() 
                      if attack_type.lower() != 'benign' and attack_type.lower() != 'normal'}

        logger.info(f'Number of benign samples: {len(benign_df)}')

        logger.info(f'Attack types: {list(attack_dfs.keys())}')

        # Define minimum attack samples for eventually balancing
        min_attack_samples = min(len(attack_df) for attack_df in attack_dfs.values())
        logger.info(f'Number of samples per attack type: {min_attack_samples}')

        # Shuffle benign_df once
        shuffled_benign_df = benign_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split benign samples into disjoint chunks, one per task
        benign_chunks = np.array_split(shuffled_benign_df, len(attack_dfs))

        # Create the output path directory if it doesn't exist
        os.makedirs(self.output().path, exist_ok=True)

        # Log the strategy
        if self.sampling_strategy == 'balanced_attacks':
            logger.info('Using the balanced attacks strategy for sampling: balance only attack samples, across all tasks')
        elif self.sampling_strategy == 'fully_balanced':
            logger.info('Using the fully balanced strategy for sampling: balance both attack samples and benign, across all tasks')
        else: # natural
            logger.info('Using the natural strategy for sampling: no balancing at all')

        # Create tasks
        for idx, (attack_type, attack_df) in enumerate(attack_dfs.items()):
            # Use the corresponding chunk of benign samples
            task_benign_df = benign_chunks[idx]

            # Apply the sampling strategy
            if self.sampling_strategy == 'balanced_attacks':
                task_attack_df = attack_df.sample(n=min_attack_samples, random_state=42) # Balance only attack samples
            elif self.sampling_strategy == 'fully_balanced':
                task_benign_df = task_benign_df.sample(n=min_attack_samples, random_state=42) # Balance benign too
                task_attack_df = attack_df.sample(n=min_attack_samples, random_state=42)
            else: # natural
                task_attack_df = attack_df # No balancing, take all available attack samples

            # Combine benign and attack samples into a single task
            task_df = pd.concat([task_benign_df, task_attack_df], ignore_index=True)

            # Compute per-task normalization
            #task_mean = task_df[columns_to_normalize].mean()
            #task_std = task_df[columns_to_normalize].std()

            # Normalize the task
            #task_df[columns_to_normalize] = (task_df[columns_to_normalize] - task_mean) / task_std

            # Save to .csv
            task_path = os.path.join(self.output().path, f'task_{idx + 1}.csv')
            
            task_df.to_csv(task_path, index=False)

            logger.info(f'Created task {idx + 1} with attack type {attack_type} with class counts: \n{task_df["attack_type"].value_counts()}')

        logger.info(f'Tasks successfully created in directory {self.output().path}')


    def output(self):
        return DirectoryTarget(get_full_rel_path(self.dataset_name, paths['tasks_rel_folder']))