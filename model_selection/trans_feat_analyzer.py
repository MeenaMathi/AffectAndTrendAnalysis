# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from feature_engineering.preprocessor import Preprocessor
from feature_engineering.feature_transformer import Feature_Transformer
from model_selection.dictionaries import Dictionaries
from data_preparation.writer import Writer

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import cross_validate

import gc
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Trans_Feat_Analyzer:
    """
    A class to analyze the 14 transformed features by training the models only with the 14 transformed features
    """

    def perform(emotion, train_tweets, y_train, task_name):
        #Select the scoring metric, depending upon the task name
        scoring = Dictionaries.scoring.get(task_name)
        if task_name == 'c':
            estimator = Dictionaries.classifier_dict
        elif task_name == 'r':
            estimator = Dictionaries.regressor_dict

        # Perform the preprocessing and feature engineering tasks
        preprocess_train_df = Preprocessor.perform(train_tweets, emotion, 'train', task_name)
        X_train = Feature_Transformer.perform(preprocess_train_df, emotion, 'train', task_name)

        # Iterate through all the estimators
        for estimator_name, estimator in estimator.items():
            print(estimator_name)
            # Default pipeline contains Feature selector + estimator
            pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                     estimator)

            scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
            print(scores)

            # Classification task
            if (task_name == 'c'):
                Writer.write_class_feat_anal_results_in_file(emotion, 'original', estimator_name, 14, scores)
                # Pipeline with resampler -SMOTE, TomekLinks, SMOTETomek
                for resampler_name, resampler in Dictionaries.resampler_dict.items():
                    # Pipeline used for resampling
                    pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                 resampler,
                                                 estimator)

                    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5,
                                            return_train_score=False)
                    print(scores)

                    Writer.write_class_feat_anal_results_in_file(emotion, resampler_name, estimator_name, 14, scores)
                    gc.collect()
            # Regression task
            elif (task_name == 'r'):
                Writer.write_reg_feat_anal_results_in_file(emotion, estimator_name, 14, scores)
                gc.collect()
