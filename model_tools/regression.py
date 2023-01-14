# Required modules
from IPython.display import clear_output
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yellowbrick

# Split
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# Imputers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

# Transformers
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

# Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV

# Regressors
from sklearn.dummy import DummyRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Yellowbrick
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError

class Tester:
    """
        A class to represent a Test.

        A Test is a custom pipeline, given a model, data and features, for solving a regression task.
    """

    def __init__(self, model, data, features, target, train_size=0.8,
                 encoding_strategy="onehot", inputing_strategy="mean",
                 scaling_strategy="standard", transforming_strategy=False):
        """
           Constructs all the necessary attributes for the test object and fits the model with given configuration.

           :param model: a scikit-learn regressor.
           :param data: a pandas dataframe.
           :param features: a list of feature strings.
           :param target: a target string.
           :param train_size: a size for a train /test split.
           :param encoding_strategy: a string for encoding strategy decision.
           :param inputing_strategy: a string for inputing strategy decision.
           :param scaling_strategy: a string for scaling strategy decision.
           :param transforming strategy: a boolean for transforming strategy decision.
        """
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.train_size = train_size
        self.encoding_strategy = encoding_strategy
        self.inputing_strategy = inputing_strategy
        self.scaling_strategy = scaling_strategy
        self.transforming_strategy = transforming_strategy
        self.numerical_features = self.data[features].select_dtypes(
                                        include=['int64', 'float64']).columns
        self.categorical_features = self.data[features].select_dtypes(
                                                    include=['object']).columns
        self.train = np.array([])
        self.test = np.array([])
        self.X_train = np.array([])
        self.X_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self._fit_model()

    def _spliter(self):
        """ 
           Splits the data according to train/test size.
        """
        self.train, self.test = train_test_split(self.data,
                                train_size=self.train_size,
                                random_state=0)
        self.X_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.X_test = self.test[self.features]
        self.y_test = self.test[self.target]

    def _numerical_transformer(self):
        """
           Returns a custom numerical transformer.
        """
        if not self.transforming_strategy:
            if self.scaling_strategy == "standard":
                return Pipeline(steps=[
                                    ('imputer', SimpleImputer(
                                     strategy=self.inputing_strategy)),
                                    ('std_scaler', StandardScaler()),
                                    ])
            if self.scaling_strategy == "robust":
                return Pipeline(steps=[
                        ('imputer', SimpleImputer(
                        strategy=self.inputing_strategy)),
                        ('std_scaler', RobustScaler()),
                        ])
        else:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(
                strategy=self.inputing_strategy)),
                ('transformer', PowerTransformer()),
                 ])

    def _categorical_transformer(self):
        """
           Returns a custom categorical transformer.
        """
        if self.encoding_strategy == "onehot":
            return Pipeline(steps=[
                ('encoder', OneHotEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        if self.encoding_strategy == "target":
            return Pipeline(steps=[
                ('encoder', TargetEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])

    def _modeler(self):
        """
           Returns a custom model pipeline.
        """
        preprocessor = ColumnTransformer(transformers=[
            ('num', self._numerical_transformer(), self.numerical_features),
            ('cat', self._categorical_transformer(), self.categorical_features),
            ])
        if not self.transforming_strategy:
            return Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
        else:
            modeling_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
            return TransformedTargetRegressor(regressor=modeling_pipeline,
                                              transformer=PowerTransformer(
                                    method='yeo-johnson', standardize=True))

    def _fit_model(self):
        """
           Fits custom model pipeline with given data and set of features.
        """
        self._spliter()
        self.fitted_model = self._modeler().fit(self.X_train, self.y_train)

    def get_fitted_model(self):
        """
           Returns the fitted custom model.
        """
        return self.fitted_model

    def get_prediction(self):
        """
           Returns the prediction.
        """
        return self.fitted_model.predict(self.X_test)

    def get_train_score(self):
        """
           Returns the train score.
        """
        return self.fitted_model.score(self.X_train, self.y_train)

    def get_test_score(self):
        """
           Returns the test score.
        """
        return self.fitted_model.score(self.X_test, self.y_test)

    def get_rmse(self):
        """
           Returns the RMSE score.
        """
        return np.sqrt(mean_squared_error(self.y_test, self.get_prediction()))

    def show_scores(self):
        """
           Prints training score, testing score and RMSE.
        """
        print("Model training score:", self.get_train_score())
        print("Model prediction score:", self.get_test_score())
        print("Model RMSE:", self.get_rmse())

    def show_residuals(self):
        """
           Shows residuals plot.
        """
        visualizer = ResidualsPlot(self._modeler())
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

    def show_prediction_errors(self):
        """
           Shows prediction error plot.
        """
        visualizer = PredictionError(self._modeler())
        visualizer.fit(self.X_train, self.y_train)
        visualizer.score(self.X_test, self.y_test)
        visualizer.show()

    def get_summary(self):
        """
           Shows summary of Test.
        """
        self.show_scores()
        self.show_residuals()
        self.show_prediction_errors()


class KFoldCrossValidator:
    """
        Class that helps to quickly cross validate
        a model with different features

    """

    def __init__(self, model, data, features, target, train_size=0.8,
                 encoding_strategy="onehot", inputing_strategy="mean",
                 scaling_strategy="standard", transforming_strategy=False,
                 cv_folds=5):
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.train_size = train_size
        self.encoding_strategy = encoding_strategy
        self.inputing_strategy = inputing_strategy
        self.scaling_strategy = scaling_strategy
        self.transforming_strategy = transforming_strategy
        self.cv_folds = cv_folds
        self.numerical_features = self.data[features].select_dtypes(
                                        include=['int64', 'float64']).columns
        self.categorical_features = self.data[features].select_dtypes(
                                                    include=['object']).columns
        self.train = np.array([])
        self.test = np.array([])
        self._cross_validate()

    def _spliter(self):
        self.train, self.test = train_test_split(
            self.data, train_size=self.train_size,
                                   random_state=0)

    def _numerical_transformer(self):
        if not self.transforming_strategy:
            if self.scaling_strategy == "standard":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', StandardScaler()),
            ])
            if self.scaling_strategy == "robust":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', RobustScaler()),
            ])
        else:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('transformer', PowerTransformer()),
                 ])

    def _categorical_transformer(self):
        if self.encoding_strategy == "onehot":
            return Pipeline(steps=[
                ('encoder', OneHotEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        if self.encoding_strategy == "target":
            return Pipeline(steps=[
                ('encoder', TargetEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])

    def _modeler(self):
        preprocessor = ColumnTransformer(transformers=[
            ('num', self._numerical_transformer(), self.numerical_features),
            ('cat', self._categorical_transformer(), self.categorical_features),
            ])
        if not self.transforming_strategy:
            return Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
        else:
            modeling_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
            return TransformedTargetRegressor(regressor=modeling_pipeline,
                                            transformer=PowerTransformer(
                                   method='yeo-johnson', standardize=True))

    def _cross_validate(self):
        self._spliter()
        self.scores = cross_validate(self._modeler(),
                            self.train[self.features],
                            self.train[self.target],
                            scoring=["neg_mean_squared_error",
                                     "neg_mean_absolute_error",
                                     "r2"],
                            cv=KFold(n_splits=self.cv_folds, shuffle=True),
                            n_jobs=-1)

    def get_average_rmse(self):
        return np.mean(np.sqrt(-self.scores['test_neg_mean_squared_error']))

    def get_sd_rmse(self):
        return np.std(np.sqrt(-self.scores['test_neg_mean_squared_error']))

    def get_average_mae(self):
        return np.mean(-self.scores['test_neg_mean_absolute_error'])

    def get_sd_mae(self):
        return np.std(-self.scores['test_neg_mean_absolute_error'])

    def get_average_r2(self):
        return np.mean(self.scores['test_r2'])

    def get_sd_r2(self):
        return np.std(self.scores['test_r2'])

    def get_summary(self):
        print("RMSE: ", self.get_average_rmse(), "(", self.get_sd_rmse(), ")")
        print("MAE: ", self.get_average_mae(), "(", self.get_sd_mae(), ")")
        print("R2: ", self.get_average_r2(), "(", self.get_sd_r2(), ")")

class LOOCrossValidator:
    """
        Class that helps to quickly cross validate
        a model with different features

    """

    def __init__(self, model, data, features, target, train_size=0.8,
                 encoding_strategy="onehot", inputing_strategy="mean",
                 scaling_strategy="standard", transforming_strategy=False):
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.train_size = train_size
        self.encoding_strategy = encoding_strategy
        self.inputing_strategy = inputing_strategy
        self.scaling_strategy = scaling_strategy
        self.transforming_strategy = transforming_strategy
        self.numerical_features = self.data[features].select_dtypes(
                                        include=['int64', 'float64']).columns
        self.categorical_features = self.data[features].select_dtypes(
                                                    include=['object']).columns
        self._cross_validate()


    def _numerical_transformer(self):
        if not self.transforming_strategy:
            if self.scaling_strategy == "standard":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', StandardScaler()),
            ])
            if self.scaling_strategy == "robust":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', RobustScaler()),
            ])
        else:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('transformer', PowerTransformer()),
                 ])

    def _categorical_transformer(self):
        if self.encoding_strategy == "onehot":
            return Pipeline(steps=[
                ('encoder', OneHotEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        if self.encoding_strategy == "target":
            return Pipeline(steps=[
                ('encoder', TargetEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])

    def _modeler(self):
        preprocessor = ColumnTransformer(transformers=[
            ('num', self._numerical_transformer(), self.numerical_features),
            ('cat', self._categorical_transformer(), self.categorical_features),
            ])
        if not self.transforming_strategy:
            return Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
        else:
            modeling_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
            return TransformedTargetRegressor(regressor=modeling_pipeline,
                                            transformer=PowerTransformer(
                                   method='yeo-johnson', standardize=True))

    def _cross_validate(self):
        self.scores = cross_validate(self._modeler(),
                            self.data[self.features],
                            self.data[self.target],
                            scoring=["neg_mean_squared_error",
                                     "neg_mean_absolute_error",
                                    ],
                            cv=LeaveOneOut(),
                            n_jobs=-1)

    def get_average_rmse(self):
        return np.mean(np.sqrt(-self.scores['test_neg_mean_squared_error']))

    def get_sd_rmse(self):
        return np.std(np.sqrt(-self.scores['test_neg_mean_squared_error']))

    def get_average_mae(self):
        return np.mean(-self.scores['test_neg_mean_absolute_error'])

    def get_sd_mae(self):
        return np.std(-self.scores['test_neg_mean_absolute_error'])

    def get_summary(self):
        print("RMSE: ", self.get_average_rmse(), "(", self.get_sd_rmse(), ")")
        print("MAE: ", self.get_average_mae(), "(", self.get_sd_mae(), ")")






class Explorer:
    """
        Class used to explore regressors with
        various settings using cross validation

    """

    def __init__(self, regressors, settings, data, target, loocv=False):
        self.regressors = regressors
        self.settings = settings
        self.data = data
        self.target = target
        self.loocv = loocv
        self.results = {}
        self.i = len(self.settings)*len(self.regressors)-1
        self.results_df = pd.DataFrame()

    def _explore_settings(self, model):
        if not self.loocv:
            results = {'model': [], 'features': [], 'inputing': [], 'encoding': [],
                       'scaling': [], 'transforming': [], 'R2': [],
                       '(R2)': [], 'RMSE': [], '(RMSE)': [],
                       'MAE': [], '(MAE)': [],
                      }
        else:
            results = {'model': [], 'features': [], 'inputing': [], 'encoding': [],
                       'scaling': [], 'transforming': [], 'RMSE': [], '(RMSE)': [],
                       'MAE': [], '(MAE)': [],
                      }
        
        for setting in self.settings:
            print(str(self.i), 'computations left...', flush=True)
            results['features'].append(list(setting[0].keys())[0])
            results['inputing'].append(setting[1])
            results['encoding'].append(setting[2])
            results['scaling'].append(setting[3])
            results['transforming'].append(setting[4])
            if not self.loocv:
                test = KFoldCrossValidator(model=model,
                                           data=self.data,
                                           features=list(
                                           setting[0].values())[0],
                                           target=self.target,
                                           inputing_strategy=setting[1],
                                           encoding_strategy=setting[2],
                                           scaling_strategy=setting[3],
                                           transforming_strategy=setting[4],
                                           cv_folds=setting[5])
                results['R2'].append(test.get_average_r2())
                results['(R2)'].append(test.get_sd_r2())
                results['RMSE'].append(test.get_average_rmse())
                results['(RMSE)'].append(test.get_sd_rmse())
                results['MAE'].append(test.get_average_mae())
                results['(MAE)'].append(test.get_sd_mae())
            else:
                test = LOOCrossValidator(model=model,
                                           data=self.data,
                                           features=list(
                                           setting[0].values())[0],
                                           target=self.target,
                                           inputing_strategy=setting[1],
                                           encoding_strategy=setting[2],
                                           scaling_strategy=setting[3],
                                           transforming_strategy=setting[4])
                results['RMSE'].append(test.get_average_rmse())
                results['(RMSE)'].append(test.get_sd_rmse())
                results['MAE'].append(test.get_average_mae())
                results['(MAE)'].append(test.get_sd_mae())
            self.i -= 1
            clear_output(wait=True)

        self.results = results

    def get_results(self):
        for name, reg in self.regressors.items():
            self._explore_settings(reg)
            names = [name]*len(self.settings)
            self.results['model'] = np.array(names)
            self.results_df = self.results_df.append(pd.DataFrame(self.results))
        return self.results_df


class RandomizedSearchCrossValidator:
    """
        Class that helps to quickly cross validate
        a model with different features

    """

    def __init__(self, model, data, features, target, train_size=0.8,
                 encoding_strategy="onehot", inputing_strategy="mean",
                 scaling_strategy="standard", transforming_strategy=False,
                 params_distribution={}):
        self.model = model
        self.data = data
        self.features = features
        self.target = target
        self.train_size = train_size
        self.encoding_strategy = encoding_strategy
        self.inputing_strategy = inputing_strategy
        self.scaling_strategy = scaling_strategy
        self.transforming_strategy = transforming_strategy
        self.params_distribution = params_distribution
        self.numerical_features = self.data[features].select_dtypes(
                                        include=['int64', 'float64']).columns
        self.categorical_features = self.data[features].select_dtypes(
                                                    include=['object']).columns
        self.train = np.array([])
        self.test = np.array([])
        self.search_cv = None
        self.cv_results = pd.DataFrame({})
        self._cross_validate()

    def _spliter(self):
        self.train, self.test = train_test_split(
            self.data, train_size=self.train_size,
                                   random_state=0)

    def _numerical_transformer(self):
        if not self.transforming_strategy:
            if self.scaling_strategy == "standard":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', StandardScaler()),
            ])
            if self.scaling_strategy == "robust":
                return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('std_scaler', RobustScaler()),
            ])
        else:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.inputing_strategy)),
                ('transformer', PowerTransformer()),
                 ])

    def _categorical_transformer(self):
        if self.encoding_strategy == "onehot":
            return Pipeline(steps=[
                ('encoder', OneHotEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        if self.encoding_strategy == "target":
            return Pipeline(steps=[
                ('encoder', TargetEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])

    def _modeler(self):
        preprocessor = ColumnTransformer(transformers=[
            ('num', self._numerical_transformer(), self.numerical_features),
            ('cat', self._categorical_transformer(), self.categorical_features),
            ])
        if not self.transforming_strategy:
            return Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
        else:
            modeling_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', self.model)
                                        ])
            return TransformedTargetRegressor(regressor=modeling_pipeline,
                                            transformer=PowerTransformer(
                                   method='yeo-johnson', standardize=True))

    def _cross_validate(self):
        self._spliter()
        self.search_cv = RandomizedSearchCV(self._modeler(),
                            self.params_distribution,
                            scoring="r2",
                            n_iter=20, random_state=0,
                            n_jobs=-1)
        self.search_cv.fit(self.train[self.features], self.train[self.target])

    def get_results(self):
        columns = [f"param_{name}" for name in self.params_distribution.keys()]
        columns += ["mean_test_error", "std_test_error"]
        self.cv_results = pd.DataFrame(self.search_cv.cv_results_)
        self.cv_results["mean_test_error"] = -self.cv_results["mean_test_score"]
        self.cv_results["std_test_error"] = self.cv_results["std_test_score"]
        self.cv_results[columns].sort_values(by="mean_test_error")
        return self.cv_results
        
