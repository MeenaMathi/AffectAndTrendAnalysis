# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


from feature_engineering.preprocessor import Preprocessor
from model_selection.dictionaries import Dictionaries
from data_preparation.writer import Writer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import pandas as pd
import gc


class Vect_Feat_Analyzer:
    """
    A class to analyze the vectorized features by training the models only with the vectorized features
    Four different vectorizers namely Count vectorizer - unigram, Tf-idf vectorizer -unigram, Count vectorizer -uni+bigram, Tf-idf vectorizer - uni+bigram are used.
    """

    def perform(emotion, train_tweets, y_train, task_name, estimator_dict, vectorizer_dict):
        # Select the scoring metric, depending upon the task name
        scoring = Dictionaries.scoring.get(task_name)

        # Perform the preprocessing and feature engineering tasks
        preprocess_train_df = Preprocessor.perform(train_tweets, emotion, 'train', task_name)

        #Iterate through all the vectorizers
        for vect_name, vectorizer in vectorizer_dict.items():
            # Convert the preprocessed text into feature vectors using vectorizer
            train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
            X_train = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())

            #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            #Iterate through all the estimators
            for estimator_name, estimator in estimator_dict.items():
                print(estimator_name, vect_name)
                # Default pipeline contains Feature selector + estimator
                pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                         estimator)

                scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
                print(scores)

                # Classification task
                if (task_name == 'c'):
                    Writer.write_class_vect_anal_results_in_file(emotion, 'original', estimator_name, vect_name, scores)
                    # Pipeline with resampler -SMOTE, TomekLinks, SMOTETomek
                    for resampler_name, resampler in Dictionaries.resampler_dict.items():
                        #Pipeline used for resampling
                        pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                                     resampler,
                                                     estimator)

                        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5, return_train_score=False)
                        print(scores)

                        Writer.write_class_vect_anal_results_in_file(emotion, resampler_name, estimator_name, vect_name,scores)
                        gc.collect()
                # Regression task
                elif(task_name == 'r'):
                    Writer.write_reg_vect_anal_results_in_file(emotion, estimator_name, vect_name, scores)
                    gc.collect()


