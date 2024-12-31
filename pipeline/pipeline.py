import luigi
import logging

# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('pipeline')


class Test(luigi.Task):
    """
    Task for testing

    Parameters:
    - hello: string
    """
    hello = luigi.Parameter()

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        logger.info(f"Hello's parameter value: {self.hello}")

        logger.info(f'Finished task {self.__class__.__name__}')