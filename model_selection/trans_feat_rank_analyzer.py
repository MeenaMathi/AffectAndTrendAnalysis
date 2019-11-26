# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from feature_engineering.preprocessor import Preprocessor
from feature_engineering.feature_transformer import Feature_Transformer
from model_selection.dictionaries import Dictionaries
from data_preparation.writer import Writer

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import cross_validate

import numpy as np
import gc
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Trans_Feat_Rank_Analyzer:
    """
    A class for analyzing the relevancy of the 14 transformed features by ranking the features based on the selection of the features using RFECV
    """

    def perform(emotion, train_tweets, y_train, task_name, estimator_dict):
        #Select the scoring metric, depending upon the task name
        scoring = Dictionaries.scoring.get(task_name)

        # Perform the preprocessing and feature engineering tasks
        preprocess_train_df = Preprocessor.perform(train_tweets, emotion, 'train', task_name)
        X_train = Feature_Transformer.perform(preprocess_train_df, emotion, 'train', task_name)

        #Iterate through all the estimators
        for estimator_name, estimator in estimator_dict.items():
            #pipeline for original data
            pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                    RFECV(estimator, step=1, cv=5, scoring = scoring, n_jobs= -1))

            scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
            print(scores)
            pipeline.fit(X_train, y_train)


            print(pipeline.steps)
            #Get number of features selected, the features selected and its ranking
            selected_features = pipeline.steps[1][1].n_features_
            feature_mask = pipeline.steps[1][1].support_
            feature_rank = pipeline.steps[1][1].ranking_

            # Classification task
            if (task_name == 'c'):
                #Get F1 scores
                cv_feature_scores = pipeline.steps[1][1].grid_scores_  # f1

                Writer.write_class_feat_rank_anal_results_in_file(emotion, 'original', estimator_name, selected_features, feature_mask, feature_rank, cv_feature_scores)
                # Pipeline with resamplers - SMOTE, TomekLinks, SMOTETomek
                for resampler_name, resampler in Dictionaries.resampler_dict.items():
                    #pipeline for resampling
                    pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                resampler,
                                                RFECV(estimator, step=1, cv=5, scoring=scoring, n_jobs=-1))

                    # Fit the pipeline with data
                    pipeline.fit(X_train, y_train)

                    print(pipeline.steps)
                    selected_features = pipeline.steps[2][1].n_features_
                    feature_mask = pipeline.steps[2][1].support_
                    feature_rank = pipeline.steps[2][1].ranking_
                    cv_feature_scores = pipeline.steps[2][1].grid_scores_  # f1

                    Writer.write_class_feat_rank_anal_results_in_file(emotion, resampler_name, estimator_name, selected_features, feature_mask, feature_rank, cv_feature_scores)
                    gc.collect()
            # Regression task
            if (task_name == 'r'):
                #Get rmse scores
                cv_feature_scores = np.sqrt(-pipeline.steps[1][1].grid_scores_)  # sqrt(-neg_mean_squared_error)

                Writer.write_reg_feat_rank_anal_results_in_file(emotion, estimator_name, selected_features, feature_mask, feature_rank, cv_feature_scores)
                gc.collect()
