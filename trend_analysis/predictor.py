from data_preparation.user_finder import User_Finder
from feature_engineering.preprocessor import Preprocessor
from feature_engineering.feature_transformer import Feature_Transformer
from model_selection.dictionaries import Dictionaries
from data_preparation.writer import Writer

import numpy as np
import gc
from pathlib import Path
import pandas as pd
from pandas import DataFrame as df
import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def check_for_file(file_path):
    #Check if the file exists, else create a new file
    if os.path.exists(file_path):
        print('File exists')
        return True
    else:
        print('File does not exist.\n Please run data_preparation > tweets_collector.py first, to collect tweets for the user')
        sys.exit(1)

def get_best_model_properties(task_name):
    """
    retrieve the best model depending upon the task name
    """
    parent_dir = Path.cwd().parent
    result_file = parent_dir.joinpath('default_results', 'score_files', 'final_best_models.csv' )
    result_dataframe = pd.DataFrame
    best_model_dict = {}
    if os.path.exists(result_file):
        result_dataframe = pd.read_csv(result_file)
    else:
        # If the file doesnt exist, exit the program with instructions
        print('\nRequired files does not exist.\n\n Please, train and assess the models first by running > modeling.py, model_assessment.py and add the files created in \'default_results\' folder')
        sys.exit(1)

    for i,emotion in Dictionaries.emo_dict.items():
        emo_results = result_dataframe[emotion + '_' + task_name].tolist()
        best_model_dict[emotion] = [emo_results[0], emo_results[1], emo_results[2], emo_results[3]]

    print(best_model_dict)
    return best_model_dict



def prepare_class_data_for_prediction(dataframe, model_dict, user_keyword, task_name):
    """
    prepare data for prediction of affects
    """
    test_tweets = dataframe.iloc[:, [0, 1, 2]]

    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')
    feature_X_user = pd.DataFrame
    emo_X_test_dict = {}


    for emotion, model_prop in model_dict.items():
        preprocessed_X_user = Preprocessor.perform(test_tweets, emotion, user_keyword, task_name)
        feature_X_user = Feature_Transformer.perform(preprocessed_X_user, emotion, user_keyword, task_name)
        vectorizer = Dictionaries.vectorizer_dict[model_prop[2]]

        #Fit transform the vectorizer with the corresponding preprocessed training data
        if os.path.exists(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl')):
            preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl'))
            train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
            print(emotion + 'TRAIN', train_vect.shape)
            train_vect_df = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())
        else:
            #If the file doesnt exist, exit the program with instructions
            print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py')
            sys.exit(1)

        # Use the same vectorizer to transform test data and then perform the feature union
        vector_X = vectorizer.transform(preprocessed_X_user['preprocessed_text'].values)
        test_vect_df = pd.DataFrame(vector_X.toarray(), columns=vectorizer.get_feature_names())
        X_test = pd.concat([test_vect_df, feature_X_user], axis=1)
        emo_X_test_dict[emotion] = X_test
        print(emotion + 'TEST', test_vect_df.shape, X_test.shape)
    return emo_X_test_dict

def prepare_reg_data_for_prediction(dataframe, model_dict, user_keyword, task_name):
    """
    prepare data for prediction of affect intensity
    """
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')

    feature_X_user,affect_dataframe, affect_index_dataframe = df, df, df
    emo_X_test_dict = {}
    affect_index_dict ={}

    for emotion, model_prop in model_dict.items():
        #Get the data with the emotion class
        if user_keyword == 'validation':
            affect_dataframe = dataframe[dataframe['Affect Dimension'] == 1]
            affect_index_list = dataframe.index[dataframe['Affect Dimension'] == 1].tolist()
        else:
            affect_dataframe = dataframe[dataframe[emotion] == 1]
            affect_index_list = dataframe.index[dataframe[emotion] == 1].tolist()
        test_tweets = affect_dataframe.iloc[:, [0, 1, 2]]

        #Perform preprocessing, feature extraction and transformation for the tweets to be predicted
        print(emotion, test_tweets.shape)
        if test_tweets.empty == False:
            preprocessed_X_user = Preprocessor.perform(test_tweets, emotion, user_keyword, task_name)
            feature_X_user = Feature_Transformer.perform(preprocessed_X_user, emotion, user_keyword, task_name)
            vectorizer = Dictionaries.vectorizer_dict[model_prop[2]]

            #Fit transform the vectorizer with the corresponding preprocessed training data
            if os.path.exists(pickle_dir.joinpath(emotion + '_r_train_preprocess_df.pkl')):
                preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_r_train_preprocess_df.pkl'))
                train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
                train_vect_df = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())
                print(emotion, 'train-shape', train_vect_df.shape, sep='\n')
            else:
                #If the file doesnt exist, exit the program with instructions
                print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py')
                sys.exit(1)

            # Use the same vectorizer to transform test data and then perform the feature union
            vector_X = vectorizer.transform(preprocessed_X_user['preprocessed_text'].values)
            test_vect_df = pd.DataFrame(vector_X.toarray(), columns=vectorizer.get_feature_names())
            X_test = pd.DataFrame(pd.concat([test_vect_df, feature_X_user], axis=1)) #####?
            emo_X_test_dict[emotion] = X_test
            affect_index_dict[emotion] = affect_index_list
        else:
            emo_X_test_dict[emotion] = pd.DataFrame
            affect_index_dict[emotion] = []

    return emo_X_test_dict, affect_index_dict

def predict_results_for_tweets(X_test, pipeline_dicts):
    """
    A method to predict affect and affect intensity of the tweets
    """
    results_dict = {}
    for emotion, pipeline in pipeline_dicts.items():
        print('Predicting for emotion: ' + emotion)
        if X_test[emotion].empty == False:
            y_pred = pipeline.predict(X_test[emotion])
            results_dict[emotion] = y_pred
        else:
            #Create array of None values with same length as that of X_test
            results_dict[emotion] = [None] * len(X_test)
    return results_dict


def get_pipelines(emo_best_model_dict, taskname, anal):
    """
    A method to get the pipelines of the best models
    """
    parent_dir = Path.cwd().parent
    task_name = 'class_' #default
    analysis = 'model_anal_' # default

    if taskname == 'r':
        task_name = 'reg_'
    if anal != 'model':
        analysis = anal + '_anal_'

    prev_name = task_name + analysis
    emo_pipeline_dict = {}
    for emotion, best_model_prop in emo_best_model_dict.items():  # dataset, classifier, vectorizer, k
        #Change k = 0 for all_in features
        if best_model_prop[0] == 'all_in':
            best_model_prop[0] = str(0)
        pipeline_path = parent_dir.joinpath('default_results', 'pipelines_' + emotion, prev_name + emotion + '_' + best_model_prop[0] + '_' + best_model_prop[1]
                                            + '_' + best_model_prop[2] + '_' + best_model_prop[3] + '.pkl')
        print(pipeline_path)
        if os.path.exists(pipeline_path):
            pipeline = pd.read_pickle(pipeline_path)
            emo_pipeline_dict[emotion] = pipeline
        else:
            # If the file doesnt exist, exit the program with instructions
            print('\nRequired files does not exist.\nPlease, train the models and select the best model for the prediction task by running model_selection > Modelling.py')
            sys.exit(1)
    print(emo_pipeline_dict)
    return emo_pipeline_dict

def merge_class_results_in_dataframe(user_dataframe, predicted_results):
    """
    A method to merge classification results in the result dataframe
    """
    user_dataframe.drop(['Affect Dimension','Intensity Score'], axis=1, inplace=True)
    user_new_df = {}
    for emotion, y_pred in predicted_results.items():
        if emotion == 'anger':
            user_dataframe.insert(2, emotion, y_pred)
        elif emotion == 'fear':
            user_dataframe.insert(3, emotion, y_pred)
        elif emotion == 'joy':
            user_dataframe.insert(4, emotion, y_pred)
        elif emotion == 'sadness':
            user_dataframe.insert(5, emotion, y_pred)
    return user_dataframe

def merge_reg_results_in_dataframe(user_data_with_affects, index_dict, predicted_reg_results):
    """
    A method to merge regression results in the result dataframe
    """
    column_index_dict = {'anger': 3, 'fear': 5, 'joy': 7, 'sadness': 9}
    for emotion, reg_results in predicted_reg_results.items():
        intensity_list = []
        index_intensity_dict = dict(zip(index_dict[emotion], reg_results))
        #Create a list of emotion intensities by index
        for i in range(0, user_data_with_affects.shape[0]):
            if i in index_dict[emotion]:
                intensity_list.append(index_intensity_dict[i])
            else:
                intensity_list.append(None)
        #Insert the list into the dataframe in the corresponding position depending upon the emotion
        user_data_with_affects.insert(column_index_dict[emotion], emotion + '_Intensitiy', intensity_list)
    return user_data_with_affects

def main():
    #Get the user name and original name
    parent_dir = Path.cwd().parent
    hardvard_csv_data = pd.read_csv(parent_dir.joinpath('documents', 'hardvardDataverseDataset.csv'))
    top_twenty_twitter_users_list = hardvard_csv_data['author'].unique().tolist()
    date_dict = {1: '2019-04-29',2: '2019-04-30',3:'2019-05-01', 4:'2019-05-02',5:'2019-05-03',6:'2019-05-04',7:'2019-05-05',8:'2019-05-06',9:'2019-05-07',10:'2019-05-08', 11: '2019-05-09'}
    user_dict = {}
    for user_nr in range(len(top_twenty_twitter_users_list)):
        original_name = User_Finder.get_user_original_name(top_twenty_twitter_users_list[user_nr])
        user_dict[user_nr + 1] = [original_name, top_twenty_twitter_users_list[user_nr]]

    # Get user input and get the tweets to be predicted for the user
    print('\nPLEASE NOTE:Currently, the prediction can be done only for the tweets in the default results folder\n')

    print(user_dict) # DO NOT REMOVE THIS PRINT STATEMENT
    selected_user = int(input("Please select the user, by entering (only) the user number: "))
    print(user_dict[selected_user][0].lower().replace(" ", ""))
    if selected_user not in range(1,21):
        print('Please run the program again and enter a valid number')
        sys.exit(1)

    print(date_dict) # DO NOT DELETE THIS PRINT STATEMENT
    selected_date = int(input('Please select the date, by choosing the number: '))
    if selected_date not in range(1,12):
        print('Please run the program again and enter a valid number')
        sys.exit(1)

    print('USERNAME: ', user_dict[selected_user][0], sep='')
    user_file = parent_dir.joinpath('default_results','pickle_files_user_data', user_dict[selected_user][0].lower().replace(" ", "") + '_' + date_dict[selected_date] + '.pkl')
    user_data = pd.DataFrame
    if(check_for_file(user_file)):
        user_data = pd.read_pickle(user_file)
        print('NUMBER OF TWEETS: ', user_data.shape[0], sep='')
        # Retrieve the best model dictionary and the corresponding pipelines for classification and regression task
        class_best_model_dict = get_best_model_properties('c')
        class_emo_pipeline_dicts = get_pipelines(class_best_model_dict, 'c', 'model')
        reg_best_model_dict = get_best_model_properties('r')
        reg_emo_pipeline_dicts = get_pipelines(reg_best_model_dict, 'r', 'model')

        # Prediction of affect class
        class_user_X_test = prepare_class_data_for_prediction(user_data, class_best_model_dict, user_dict[selected_user][0].lower().replace(" ", "")
                                                              + '_' + date_dict[selected_date], 'c')
        predicted_class_results = predict_results_for_tweets(class_user_X_test, class_emo_pipeline_dicts)
        user_data_with_affects = merge_class_results_in_dataframe(user_data, predicted_class_results)
        #Writer.write_the_classification_results(user_data_with_affects)
        gc.collect()

        # Prediction of affect intensity
        if (user_data_with_affects.empty == False):
            reg_user_X_test_dict, affect_index_dict = prepare_reg_data_for_prediction(user_data_with_affects, reg_best_model_dict,  user_dict[selected_user][0].lower().replace(" ", "")  + '_' + date_dict[selected_date], 'r')
            if len(reg_user_X_test_dict) > 0:
                predicted_reg_results = predict_results_for_tweets(reg_user_X_test_dict, reg_emo_pipeline_dicts)
                user_data_with_affects_and_intensity = merge_reg_results_in_dataframe(user_data_with_affects, affect_index_dict, predicted_reg_results)
                print(user_data_with_affects_and_intensity)
                gc.collect()
                Writer.write_the_final_predicted_results(user_data_with_affects_and_intensity, user_dict[selected_user][0].lower().replace(" ", "")  , date_dict[selected_date])
            else:
                Writer.write_the_final_predicted_results(user_data_with_affects, user_dict[selected_user][0].lower().replace(" ", "") , date_dict[selected_date])
        else:
            Writer.write_the_final_predicted_results(user_data_with_affects, user_dict[selected_user][0].lower().replace(" ", "") , date_dict[selected_date])


# init
if __name__ == '__main__':
    main()