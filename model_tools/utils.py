from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np


class Regressor:
    '''
        A class that builds a Regressor object    
    '''
    def __init__(self, regressor=DummyRegressor(), name='dummy_regressor'):
        """
           Constructs all the necessary attributes for the Data object.
            :param regressor: a scikit-learn regressor
            :param name: a string name of the regressor
        """
        self.regressor = regressor
        self.name = name

class Data:
    """
        A class that builds a Data object
    """
    def __init__(self, dataframe=DataFrame(), features=[], target="", split_size=0.2):
        """
           Constructs all the necessary attributes for the Data object.
           :param dataframe: a pandas.DataFrame object
           :param features: feature's names as a list of strings
           :param target: target's name as a string
        """
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.split_size = split_size
        self.train = np.array([])
        self.test = np.array([])
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self._spliter()

    def _spliter(self):
        """ 
           Splits the data according to train/test size.
        """
        self.train, self.test = train_test_split(self.dataframe,
                                train_size=self.split_size,
                                random_state=0)
        self.X_train, self.y_train  = self.train[self.features], self.train[self.target]
        self.X_test, self.y_test = self.test[self.features], self.test[self.target]

class Setting:
    """
        A class that builds a Setting object
    """
    def __init__(self, name="setting0", encoding_strategy="one-hot", inputing_strategy="mean", scaling_strategy='standard', transforming_strategy=False):
        """
           Constructs all the necessary attributes for the Setting object.
           :param encoding_strategy: a string for encoding strategy decision.
           :param inputing_strategy: a string for inputing strategy decision.
           :param scaling_strategy: a string for scaling strategy decision.
           :param transforming strategy: a boolean for transforming strategy decision.
        """
        self.name = name
        self.encoding_strategy = encoding_strategy
        self.inputing_strategy = inputing_strategy
        self.scaling_strategy = scaling_strategy
        self.transforming_strategy = transforming_strategy
