from model_selection.score_util import Score_Util
from data_preparation.sem_eval_task1_data import SemEval_Task1_Data
from data_preparation.data_manager import Data_Manager
from data_preparation.writer import Writer
from model_selection.dictionaries import Dictionaries
from feature_engineering.preprocessor import Preprocessor
from feature_engineering.feature_transformer import Feature_Transformer

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import cross_val_predict

from pathlib import Path
import sys
import os
import pandas as pd
import pickle as pickle
import numpy as np
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
"""
This class is used to evaluate and assess the best models using the test data.
"""

def prepare_data_for_prediction(emotion, dataframe, model_prop, user_keyword, task_name):
    """
    A method to prepare the data suitable for prediction phase
    """
    test_tweets = dataframe.iloc[:, [0, 1, 2]]

    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')
    feature_X_user = pd.DataFrame


    preprocessed_X_user = Preprocessor.perform(test_tweets, emotion, user_keyword, task_name)
    feature_X_user = Feature_Transformer.perform(preprocessed_X_user, emotion, user_keyword, task_name)
    vectorizer = Dictionaries.vectorizer_dict[model_prop[2]]

    #Fit transform the vectorizer with the corresponding preprocessed training data
    if os.path.exists(pickle_dir.joinpath(emotion + '_' + task_name + '_train_preprocess_df.pkl')):
        preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_' + task_name + '_train_preprocess_df.pkl'))
        train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
        print(emotion + 'TRAIN', train_vect.shape)
        train_vect_df = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())
    else:
        #If the file doesnt exist, exit the program with instructions
        print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py and add the files created in \'default_results\' folder')
        sys.exit(1)

    # Use the same vectorizer to transform test data and then perform the feature union
    vector_X = vectorizer.transform(preprocessed_X_user['preprocessed_text'].values)
    test_vect_df = pd.DataFrame(vector_X.toarray(), columns=vectorizer.get_feature_names())
    X_test = pd.concat([test_vect_df, feature_X_user], axis=1)

    #print(emotion + 'TEST', test_vect_df.shape, X_test.shape)
    return X_test

def get_pipelines(emo_best_model_dict, taskname, anal):
    """
    A method to retrieve the pipelines of best models
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
    for name, best_model_prop in emo_best_model_dict.items():  # dataset, classifier, vectorizer, k, scores
        str = name.split('_')
        emotion = str[0]
        #Change k = 0 for all_in features
        if best_model_prop[3] == 'all_in':
            best_model_prop[3] = str(0)
        pipeline_path = parent_dir.joinpath('default_results', 'pipelines_' + emotion, prev_name + emotion + '_' + best_model_prop[0] + '_' + best_model_prop[1]
                                            + '_' + best_model_prop[2] + '_' + best_model_prop[3] + '.pkl')

        if os.path.exists(pipeline_path):
            pipeline = pd.read_pickle(pipeline_path)
            emo_pipeline_dict[emotion] = pipeline
        else:
            # If the file doesnt exist, exit the program with instructions
            print('\nRequired files does not exist.\nPlease, train the models and select the best model for the prediction task by running model_selection > Modelling.py')
            sys.exit(1)

    return emo_pipeline_dict

def calculate_twenty_percent(emo_size,total_size):
    """
    A Method ot calculate 20% using the available number of data samples
    """
    # Total number of data = 80%
    # 20 % of data = total number of data * 20 /80 = Number of data * 0.25
    other_size = total_size - emo_size
    emo_20 = int(emo_size * 0.25)
    other_20 = int(other_size * 0.25)
    return emo_20, other_20

def prepare_final_class_test_data(emotion, test_df, emo_size, other_size):
    """
    A method to prepare test data used for evaluation and model assessment
    """
    emo_data = test_df[test_df['Affect Dimension'] == emotion].head(emo_size)
    other_data = test_df[test_df['Affect Dimension'] != emotion].head(other_size)
    new_test_df = pd.concat([emo_data, other_data])
    print(new_test_df.shape, new_test_df['Affect Dimension'].value_counts(), sep = '\n')
    return new_test_df

def prepare_X_y_data_for_task(task_name):
    """
    A method to split X and y depending upon the task
    """
    parent_dir = Path.cwd().parent
    # print(parent_dir)
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')
    pickle_file_name = pickle_dir.joinpath('emo_' + task_name + '_val_data_dict.pkl')

    # Check the existence of the file and if the file is not empty
    if os.path.exists(pickle_file_name) and os.path.getsize(pickle_file_name) > 0:
        with open(pickle_file_name, 'rb') as in_file:
            emotion_X_y_dict = pickle.load(in_file)
            return emotion_X_y_dict
    else:
        # Create a directory, if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        emo_X_y_dict = {}
        for i, emotion in Dictionaries.emo_dict.items():
            if task_name == 'c':
                #CLASSIFiCATION TASK
                class_train_data = SemEval_Task1_Data.merge_training_data('c')
                class_test_data = SemEval_Task1_Data.merge_test_data('c')
                emotion_class_train_data = Data_Manager.get_train_data_for_emotion(emotion, class_train_data, 'c', 'train')
                emotion_class_test_data = Data_Manager.get_test_data_for_emotion(emotion, class_test_data, 'c', 'test')
                print(emotion_class_train_data.shape(), emotion_class_test_data.shape(), emotion_class_train_data[emotion_class_train_data['Affect Dimension'] == emotion].shape[0])
                emo_class_20, other_class_20 = calculate_twenty_percent(emotion_class_train_data[emotion_class_train_data['Affect Dimension'] == emotion].shape[0], emotion_class_train_data.shape[0])
                print(emo_class_20, other_class_20)
                final_emo_class_test_data = prepare_final_class_test_data(emotion, emotion_class_test_data, emo_class_20, other_class_20)
                X_train, y_train = Data_Manager.split_data_into_X_and_y(emotion, emotion_class_train_data, 'c')
                X_test, y_test = Data_Manager.split_data_into_X_and_y(emotion, final_emo_class_test_data, 'c')
                print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                emo_X_y_dict[emotion] = [X_train, X_test, y_train, y_test]
            elif(task_name == 'r'):
                reg_train_data = SemEval_Task1_Data.merge_training_data('r')
                reg_test_data = SemEval_Task1_Data.merge_test_data('r')
                emotion_reg_train_data = Data_Manager.get_train_data_for_emotion(emotion, reg_train_data, 'r','train')
                emotion_reg_test_data = Data_Manager.get_test_data_for_emotion(emotion, reg_test_data, 'r', 'test')
                print(emotion_reg_train_data.shape, emotion_reg_test_data.shape)
                emo_reg_20 = int(emotion_reg_train_data.shape[0]*0.25) # 20% data calculation
                print(emo_reg_20)
                emo_reg_test_data = emotion_reg_test_data.head(emo_reg_20)
                X_train, y_train = Data_Manager.split_data_into_X_and_y(emotion, emotion_reg_train_data, 'r')
                X_test, y_test = Data_Manager.split_data_into_X_and_y(emotion, emo_reg_test_data, 'r')
                print(X_train.shape, X_test.shape, y_train.shape)
                emo_X_y_dict[emotion] = [X_train, X_test, y_train, y_test]

        with open(pickle_file_name, 'wb') as outfile:
            pickle.dump(emo_X_y_dict, outfile)

        return emo_X_y_dict

def train_the_best_models_again(model_properties):
    """
    A method to train the classification best models using original and resampled dataset again
    """
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')
    results_dir = parent_dir.joinpath('default_results', 'score_files')
    best_scores = {}
    scores_dict = {}
    for i,emotion in Dictionaries.emo_dict.items():
        best_model_original = model_properties[emotion + '_class_original' ]
        best_model_resampled = model_properties[emotion + '_class_resampled']

        #Fit transform the vectorizer with the corresponding preprocessed training data
        if os.path.exists(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl')):
            preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl'))
            trans_feat_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_c_train_feat_transform_df.pkl'))

            #Use the corresponding vectorizer from the model properties to vectorize
            train_vect_original = Dictionaries.vectorizer_dict[best_model_original[2]].fit_transform(preprocess_train_df['preprocessed_text'].values)
            train_vect_df_original = pd.DataFrame(train_vect_original.toarray(), columns=Dictionaries.vectorizer_dict[best_model_original[2]].get_feature_names())
            train_vect_resampled = Dictionaries.vectorizer_dict[best_model_resampled[2]].fit_transform(preprocess_train_df['preprocessed_text'].values)
            train_vect_df_resampled = pd.DataFrame(train_vect_resampled.toarray(),columns=Dictionaries.vectorizer_dict[best_model_resampled[2]].get_feature_names())

            #merge vectorized features and transformed features
            X_train_original = pd.DataFrame(pd.concat([train_vect_df_original, trans_feat_train_df], axis=1))
            X_train_resampled = pd.DataFrame(pd.concat([train_vect_df_resampled, trans_feat_train_df], axis=1))
            y_train = preprocess_train_df['Affect Dimension'].astype('category').cat.rename_categories({emotion: 1, 'other': 0})

            # pipeline for original dataset
            pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                     SelectKBest(chi2, k= int(best_model_original[3])),
                                     Dictionaries.classifier_dict[best_model_original[1]])
            if best_model_original[3] == 0:
                pipeline = make_pipeline(MinMaxScaler(feature_range=(0, 1), copy=True),
                                         Dictionaries.classifier_dict[best_model_original[1]])

            y_pred_original = cross_val_predict(pipeline, X_train_original, y_train, cv=5)

            # pipeline for resampled dataset
            pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                         SelectKBest(chi2, k= int(best_model_resampled[3])),
                                         Dictionaries.resampler_dict[best_model_resampled[0]],
                                         Dictionaries.classifier_dict[best_model_resampled[1]])
            if best_model_resampled[3] == 0:
                pipeline = make_pipeline_imb(MinMaxScaler(feature_range=(0, 1), copy=True),
                                             Dictionaries.resampler_dict[best_model_resampled[0]],
                                             Dictionaries.classifier_dict[best_model_resampled[1]])

            y_pred_resampled = cross_val_predict(pipeline, X_train_resampled, y_train, cv=5)

            scores_original = classification_report( y_train, y_pred_original , labels=[1, 0], output_dict=True )
            accuracy_original = accuracy_score(y_train, y_pred_original)
            scores_resampled = classification_report(y_train, y_pred_resampled, labels=[1, 0], output_dict=True )
            accuracy_resampled = accuracy_score(y_train, y_pred_resampled)

            print(scores_original, scores_resampled)
            scores_dict[emotion + 'original'] = [scores_original, accuracy_original]
            scores_dict[emotion + 'resampled'] = [scores_resampled, accuracy_resampled]
            emo_f1_original = scores_original['1']['f1-score']
            avg_f1_original = scores_original['macro avg']['f1-score']
            emo_f1_resampled = scores_resampled['1']['f1-score']
            avg_f1_resampled = scores_resampled['macro avg']['f1-score']

            #Add the results needed for analysis to the dict
            best_scores[emotion] = [avg_f1_original, emo_f1_original, accuracy_original, avg_f1_resampled, emo_f1_resampled, accuracy_resampled]
        else:
            #If the file doesnt exist, exit the program with instructions
            print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py and add the files created in \'default_results\' folder')
            sys.exit(1)

    # store the classification report and accuracy of both the models
    with open(results_dir.joinpath('best_class_both_model_scores.pkl'), 'wb') as outfile:
        pickle.dump(scores_dict, outfile)

    return best_scores

def analyze_scores_and_find_class_best_model(score_dict, best_model_dict):
    """
    A method to analyze the emotion F1 scores and select the best classification model based on the highest score
    """
    parent_dir = Path.cwd().parent
    input_file_name = parent_dir.joinpath('default_results','score_files','class_final_best_model_score.pkl')

    # Check the existence of the file and if the file is not empty
    if os.path.exists(input_file_name) and os.path.getsize(input_file_name) > 0:
        best_model_prop = pd.read_pickle(input_file_name)
        return best_model_prop
    else:
        # Create a directory for storing pickle files, if it doesn't exist
        if not os.path.exists(input_file_name):
            os.makedirs(input_file_name)

        best_model_prop = {}
        for i, emotion in Dictionaries.emo_dict.items():
            score_list = score_dict[emotion]
            emo_f1_original = score_list[1]
            emo_f1_resampled = score_list[4]
            #select the model with highest emotion class F1
            if emo_f1_resampled >= emo_f1_original:
                best_model_prop[emotion] = best_model_dict[emotion + '_class_resampled']
            else:
                best_model_prop[emotion] = best_model_dict[emotion + '_class_original']

        with open(parent_dir.joinpath('score_files','class_final_best_model_score.pkl'), 'wb') as infile:
            pickle.dump(best_model_prop, infile, protocol=pickle.HIGHEST_PROTOCOL)
        return best_model_prop


def main():
    # Retrieve the best model dictionary and the corresponding pipelines for classification and regression task
    best_model_dict, merged_df, row_index_dict = Score_Util.select_the_best_model('classification_')
    print('BEST CLASSIFICATION MODELS: ' , best_model_dict, sep='\n')
    score_list = train_the_best_models_again(best_model_dict)
    class_best_model_prop = analyze_scores_and_find_class_best_model(score_list, best_model_dict)
    print('FINAL BEST CLASSIFICATION MODEL: ', class_best_model_prop, sep='\n')
    class_emo_pipeline_dicts = get_pipelines(class_best_model_prop, 'c', 'model')
    reg_best_model_properties = Score_Util.select_the_best_model('regression_')
    reg_best_model_dict = reg_best_model_properties[0]
    reg_emo_pipeline_dicts = get_pipelines(reg_best_model_dict, 'r', 'model')
    print('FINAL BEST REGRESSION MODEL: ', reg_best_model_properties, sep='\n')
    print('CLASSIFICATION AND REGRESSION PIPELINES')
    print(class_emo_pipeline_dicts)
    print(reg_emo_pipeline_dicts)

    # Store the final best models for prediction task
    Writer.write_the_best_model_properties(class_best_model_prop, reg_best_model_dict)

    class_emo_X_y_dict = prepare_X_y_data_for_task('c')
    reg_emo_X_y_dict = prepare_X_y_data_for_task('r')

    # predict and evaluate for the test data
    for i, emotion in Dictionaries.emo_dict.items():
        #Classification
        #Prepare the data for prediction
        X_y_class_data = class_emo_X_y_dict[emotion]
        X_class_train, X_class_test, y_class_train, y_class_test = tuple([X_y_class_data[0], X_y_class_data[1], X_y_class_data[2], X_y_class_data[3]])
        X_class_test_new = prepare_data_for_prediction(emotion, X_class_test, class_best_model_prop[emotion], 'model_assessment', 'c')

        #Predict the class for the test data using the best pipeline
        y_class_pred = np.array(class_emo_pipeline_dicts[emotion].predict(X_class_test_new))
        prec, recall, f1, support = precision_recall_fscore_support(y_class_test, y_class_pred, average=None, labels=[1, 0])
        avg_prec, avg_recall, avg_f1, avg_support = precision_recall_fscore_support(y_class_test, y_class_pred, average='macro',labels=[1, 0])
        accuracy = accuracy_score(y_class_test, y_class_pred)
        Writer.write_class_results_for_test_data(emotion, class_best_model_prop[emotion], prec, recall, accuracy, f1, avg_prec, avg_recall, avg_f1)

        #Regression
        #Prepare the data for prediction
        X_y_reg_data = reg_emo_X_y_dict[emotion]
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = tuple([X_y_reg_data[0], X_y_reg_data[1], X_y_reg_data[2], X_y_reg_data[3]])
        #Change emotion column into numeric value 1
        temp_X_reg_test = X_reg_test.replace({'Affect Dimension': emotion}, 1)
        X_reg_test_new = prepare_data_for_prediction(emotion, temp_X_reg_test, reg_best_model_dict[emotion + '_reg_original'], 'model_assessment', 'r')

        # Predict the intensity for the test data using the best pipeline
        y_reg_pred = np.array(reg_emo_pipeline_dicts[emotion].predict(X_reg_test_new))
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        Writer.write_reg_results_for_test_data(emotion, reg_best_model_dict[emotion + '_reg_original'], mae, rmse)


# init
if __name__ == '__main__':
    main()
