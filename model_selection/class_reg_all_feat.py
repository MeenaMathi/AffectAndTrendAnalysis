# import warnings filter
from warnings import simplefilter
# ignore all warnings
simplefilter(action='ignore', category=FutureWarning)

from feature_engineering.preprocessor import Preprocessor
from feature_engineering.feature_transformer import Feature_Transformer
from model_selection.dictionaries import Dictionaries
from data_preparation.writer import Writer

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import cross_validate
import pickle as pick
import pandas as pd
import gc
from pathlib import Path


class Class_Reg_All_Feat:
    """
    This class performs the training phase depending upon the emotion, task and the selected number of features
    """
    def perform(emotion, train_tweets, y_train, task_name, k, estimator_dict, vectorizer_dict):
        parent_dir = Path.cwd().parent
        pipelines_dir = parent_dir.joinpath('new_results','pipelines_' + emotion)
        Writer.check_for_directory(pipelines_dir)
        
        #Select the scoring metric, depending upon the task name
        scoring = Dictionaries.scoring.get(task_name)

        # Perform the preprocessing and feature engineering tasks
        preprocess_train_df = Preprocessor.perform(train_tweets, emotion, 'train',task_name)
        trans_feat_train_df = Feature_Transformer.perform(preprocess_train_df, emotion, 'train',task_name)

        #Iterate through all the vectorizers
        for vect_name, vectorizer in vectorizer_dict.items():
            # Convert the preprocessed text into feature vectors using vectorizer
            train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
            train_vect_df = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())
            print('TRAIN_VECTORIZED')
            print(train_vect_df.shape)

            # Final training data: Merge Feature vector columns with transformed features columns -> X_train, X_test
            X_train = pd.concat([train_vect_df, trans_feat_train_df], axis = 1)
            print('X_train, y_train with vector features + features transformed')
            print(X_train.shape, y_train.shape)

            #Iterate through all the estimators
            for estimator_name, estimator in estimator_dict.items():
                ########################### CLASSIFICATION ##################################
                if (task_name == 'c'):

                    # Default pipeline contains Feature selector + estimator, where as if k = 0(all_in), the pipeline doesnot contain the Feature selector
                    pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                            SelectKBest(chi2, k=k),
                                            estimator)
                    if k == 0:
                        pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                estimator)

                    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
                    print(scores)

                    # Fit the same pipeline for train data and predict the results for X_test to find the test scores
                    pipeline.fit(X_train, y_train)
                    # Store the pipeline as pickle files
                    with open(pipelines_dir.joinpath('class_model_anal_' + emotion + '_original_' + estimator_name + '_' + vect_name + '_' + str(k) + '.pkl') , 'wb') as infile:
                        pick.dump(pipeline, infile, pick.HIGHEST_PROTOCOL)
                        infile.close()

                    Writer.write_class_model_anal_results_in_file(emotion, 'original', estimator_name, vect_name, k, scores)

                    ##################################### CLASSIFICATION RESAMPLING ###################################################################

                    # Pipeline with resampler -SMOTE, TomekLinks, SMOTETomek
                    for resampler_name, resampler in Dictionaries.resampler_dict.items():
                        print(estimator_name, vect_name, resampler_name)

                        pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                    SelectKBest(chi2, k=k),
                                                    resampler,
                                                    estimator)
                        if k == 0:
                            pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                        resampler,
                                                        estimator)

                        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
                        print(scores)

                        # Fit the same pipeline for train data and predict the results for fixed X_test to find the test scores
                        pipeline.fit( X_train, y_train)
                        # Store the pipeline as pickle files
                        with open(pipelines_dir.joinpath('class_model_anal_' + emotion + '_' + resampler_name + '_' + estimator_name + '_' + vect_name + '_' + str(k) + '.pkl'), 'wb') as infile:
                            pick.dump(pipeline, infile, pick.HIGHEST_PROTOCOL)
                            infile.close()

                        Writer.write_class_model_anal_results_in_file(emotion, resampler_name, estimator_name, vect_name, k, scores)
                        gc.collect()


                ######################## REGRESSION #############################################
                elif(task_name == 'r'):
                    # Default pipeline contains Feature selector + estimator, where as if k = 0, the pipeline doesnot contain the Feature selector
                    pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                            SelectKBest(f_regression, k=k),
                                            estimator)
                    if k == 0:
                        pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                 estimator)

                    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
                    print(scores)

                    # Fit the same pipeline for train data and predict the results for fixed X_test to find the test scores
                    pipeline.fit(X_train, y_train)
                    # Store the pipeline as pickle files
                    with open( pipelines_dir.joinpath('reg_model_anal_'+ emotion + '_original_' + estimator_name + '_' + vect_name + '_' + str(k) + '.pkl'), 'wb') as infile:
                        pick.dump(pipeline, infile, pick.HIGHEST_PROTOCOL)
                        infile.close()

                    Writer.write_reg_model_anal_results_in_file(emotion, estimator_name, vect_name, k, scores)
                    gc.collect()


