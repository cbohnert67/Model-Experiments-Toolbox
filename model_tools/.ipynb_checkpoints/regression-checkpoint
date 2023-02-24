from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
import numpy as np

class CustomPipeline:
    """
        A class representing a CustomPipeline.

        A CustomPipeline is built with the Setting object and the Regressor is trained using the Data object. 
        Performance scores are computed such as RMSE and MAE and a Yellowbrick visualizer is showed.
    """
    def __init__(self, regressor, data, setting):
        """
           Constructs all the necessary attributes for the CustomPipeline object.

           :param regressor: a regressor object.
           :param data: a Data object.
           :param setting: a Setting object
        """
        self.regressor = regressor
        self.data = data
        self.setting = setting
        
    def _numerical_transformer(self):
        """
           Returns a custom numerical transformer.
        """
        if self.setting.transforming_strategy in ["yeo-johnson", "box-cox"]:
            return Pipeline(steps=[
                ('imputer', SimpleImputer(
                strategy=self.setting.inputing_strategy)),
                ('transformer', PowerTransformer(method=self.setting.transforming_strategy, standardize=True)),
                 ])
        else:
            if self.setting.scaling_strategy == "standard":
                return Pipeline(steps=[
                                    ('imputer', SimpleImputer(
                                     strategy=self.setting.inputing_strategy)),
                                    ('scaler', StandardScaler()),
                                    ])
            if self.setting.scaling_strategy == "robust":
                return Pipeline(steps=[
                        ('imputer', SimpleImputer(
                        strategy=self.setting.num_inputing_strategy)),
                        ('scaler', RobustScaler()),
                        ])  
            if self.setting.scaling_strategy == "minmax":
                return Pipeline(steps=[
                        ('imputer', SimpleImputer(
                        strategy=self.setting.num_inputing_strategy)),
                        ('scaler', MinMaxScaler()),
                        ])              
    def _categorical_transformer(self):
        """
           Returns a custom categorical transformer.
        """
        if self.setting.encoding_strategy == "onehot":
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.setting.cat_inputing_strategy)),
                ('encoder', OneHotEncoder()),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        if self.setting.encoding_strategy == "target":
            return Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.setting.cat_inputing_strategy)),
                ('encoder', TargetEncoder()),
                ('std_scaler', StandardScaler(with_mean=False)),
                ])
        def _modeler(self):
            """
            Returns a custom model pipeline.
            """
            features = self.data.features
            preprocessor = ColumnTransformer(transformers=[
                ('num', self._numerical_transformer(), features.select_dtypes(
                                        include=['int64', 'float64']).columns),
                ('cat', self._categorical_transformer(), features.select_dtypes(
                                                    include=['object']).columns),
                ])
            if self.transforming_strategy in ["yeo-johnson", "box-cox"]:
                modeling_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', self.model)
                                            ])
                return TransformedTargetRegressor(regressor=modeling_pipeline,
                                                transformer=PowerTransformer(
                                        method='yeo-johnson', standardize=True))
            else:
                return Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', self.model)
                                            ])
        def _fit_model(self):
            """
            Fits custom model pipeline with given data and set of features.
            """
            self.fit_model = self._modeler().fit(self.data.X_train, self.data.y_train)
        def get_fit_model(self):
            """
            Returns the fit custom model.
            """
            return self.fit_model

        def get_prediction(self):
            """
            Returns the prediction.
            """
            return self.fit_model.predict(self.data.X_test)

        def get_train_score(self):
            """
            Returns the train score.
            """
            return self.fit_model.score(self.data.X_train, self.data.y_train)

        def get_test_score(self):
            """
            Returns the test score.
            """
            return self.fit_model.score(self.data.X_test, self.data.y_test)

        def get_rmse(self):
            """
            Returns the RMSE score.
            """
            return np.sqrt(mean_squared_error(self.data.y_test, self.get_prediction()))
        
        def get_mae(self):
            """
            Returns the MAE score.
            """
            return mean_absolute_error(self.data.y_test, self.get_prediction())

        def show_scores(self):
            """
            Prints training score, testing score and RMSE.
            """
            print("Model training score:", self.get_train_score())
            print("Model prediction score:", self.get_test_score())
            print("Model RMSE:", self.get_rmse())
            print("Model MAE:", self.get_mae())

        def show_residuals(self):
            """
            Shows residuals plot.
            """
            visualizer = ResidualsPlot(self._modeler())
            visualizer.fit(self.data.X_train, self.data.y_train)
            visualizer.score(self.data.X_test, self.data.y_test)
            visualizer.show()

        def show_prediction_errors(self):
            """
            Shows prediction error plot.
            """
            visualizer = PredictionError(self._modeler())
            visualizer.fit(self.data.X_train, self.data.y_train)
            visualizer.score(self.data.X_test, self.data.y_test)
            visualizer.show()

        def get_summary(self):
            """
            Shows summary of Test.
            """
            self.show_scores()
            self.show_residuals()
            self.show_prediction_errors()
        
