import luigi
import logging
import os
import glob
import pandas as pd


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('data-prep-pipeline')

# Get configuration file
config = luigi.configuration.get_config()

# Dict that contains the parameters configurated in luigi.cfg
paths = {
    'datasets_folder': config.get('Paths', 'datasets_folder'),
    'merged_rel_filename': config.get('Paths', 'merged_rel_filename'), 
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
    TODO
    """
    
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension

    # Luigi's complete method override
    def complete(self):
        # True if there's at least a file with the given extension in the given path
        return len(glob.glob(f"{self.path}/*{self.extension}")) > 0

    # Luigi's complete method override
    def exists(self):
        return len(glob.glob(f"{self.path}/*{self.extension}")) > 0



class MergeFiles(luigi.Task):
    """
    TODO
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
        if self.input_file_extension.lower() == ".parquet":
            data_frames = [pd.read_parquet(file) for file in matching_files]
        else: # .csv
            data_frames = [pd.read_csv(file) for file in matching_files]
        
        merged_df = pd.concat(data_frames, ignore_index=True)

        # Create the output path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Save to .parquet
        merged_df.to_parquet(self.output().path)

        logger.info(f'Merged {len(matching_files)} files into the parquet file {self.output().path}')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.dataset_name, 
                                                   paths['merged_rel_filename']))