import pandas as pd
from pandas import DataFrame as df
import os
import pickle
from pathlib import Path


class Data_Manager:
    """
    A class that prepares suitable data depending upon the task and emotion.
    """

    def get_train_data_for_emotion(emotion, dataframe, task_name, data_flag):
        """
        A Method to prepare training data for the emotion depending upon the task.
        """

        parent_dir = Path.cwd().parent
        pickle_dir = parent_dir.joinpath('new_results','pickle_files_feat_eng')
        pickle_file_name = pickle_dir.joinpath('emo_' + task_name + '_' + data_flag + '_data_dict.txt')

        # Check the existence of the file and if the file is not empty
        if os.path.exists(pickle_file_name) and os.path.getsize(pickle_file_name) > 0:
            with open(pickle_file_name, 'rb') as in_file:
                emotion_data_dict = pickle.load(in_file)
                return emotion_data_dict.get(emotion)
        else:
            # Create a directory, if it doesn't exist
            if not os.path.exists(pickle_dir):
                os.makedirs(pickle_dir)

            anger_train_data, fear_train_data, joy_train_data, sadness_train_data = df, df, df, df

            # Training data with emotion X in classification and regression task respectively
            only_anger_train = dataframe[dataframe['Affect Dimension'] == 'anger']  # 3542, 2089
            only_fear_train = dataframe[dataframe['Affect Dimension'] == 'fear']  # 2879, 2641
            only_joy_train = dataframe[dataframe['Affect Dimension'] == 'joy']  # 3228, 1904
            only_sadness_train = dataframe[dataframe['Affect Dimension'] == 'sadness']  # 3156, 1930

            # Classification task
            if task_name.casefold() == 'c':

                none_data_train = dataframe[dataframe['Affect Dimension'] == 'none']  # All the 11 emotions are 0: 218
                other_data_train = dataframe[dataframe['Affect Dimension'] == 'other']  # All the 4 emotions are 0, but not all 11: 519

                # Convert the emotion into 'other' for each emotion
                none_to_other_data = none_data_train.replace('none', 'other')
                anger_to_other_data = only_anger_train.replace('anger', 'other')
                fear_to_other_data = only_fear_train.replace('fear', 'other')
                joy_to_other_data = only_joy_train.replace('joy', 'other')
                sadness_to_other_data = only_sadness_train.replace('sadness', 'other')

                # Remove duplicates, keeping the corresponding Emotion X = 7801
                # Anger: 3542 Vs 4259, Fear: 2879 Vs 4922, Joy: 3228 Vs 4573, sadness: 3156 Vs 4645
                anger_train_data = pd.concat([only_anger_train, none_to_other_data, other_data_train, fear_to_other_data, joy_to_other_data,
                                        sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                fear_train_data = pd.concat([only_fear_train, none_to_other_data, other_data_train, anger_to_other_data, joy_to_other_data,
                                        sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                joy_train_data = pd.concat([only_joy_train, none_to_other_data, other_data_train, anger_to_other_data, fear_to_other_data,
                                        sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                sadness_train_data = pd.concat([only_sadness_train, none_to_other_data, other_data_train, anger_to_other_data, fear_to_other_data,
                                        joy_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')

            # Regression task
            elif task_name.casefold() == 'r':
                # Do nothing. Reassign to emotion_data
                anger_train_data = only_anger_train #2089
                fear_train_data = only_fear_train #2641
                joy_train_data = only_joy_train #1906
                sadness_train_data = only_sadness_train #1930


            emotion_data_dict = {
                'anger': anger_train_data,
                'fear': fear_train_data,
                'joy': joy_train_data,
                'sadness': sadness_train_data
            }

            with open(pickle_file_name, 'wb') as outfile:
                pickle.dump(emotion_data_dict, outfile)
            return emotion_data_dict.get(emotion)

    def get_test_data_for_emotion(emotion, test_gold_df, task_name, data_flag):
        """
        A Method to prepare test data for the emotion depending upon the task.
        """
        parent_dir = Path.cwd().parent
        pickle_dir = parent_dir.joinpath('new_results','pickle_files_feat_eng')
        pickle_file_name = pickle_dir.joinpath('emo_' + task_name + '_' + data_flag + '_data_dict.txt')

        # Check the existence of the file and if the file is not empty
        if os.path.exists(pickle_file_name) and os.path.getsize(pickle_file_name) > 0:
            with open(pickle_file_name, 'rb') as in_file:
                emotion_data_dict = pickle.load(in_file)
                return emotion_data_dict.get(emotion)
        else:
            # Create a directory, if it doesn't exist
            if not os.path.exists(pickle_dir):
                os.makedirs(pickle_dir)

            anger_test_data, fear_test_data, joy_test_data, sadness_test_data = df, df, df, df

            # Test data with emotion X in classification and regression task respectively
            only_anger_test = test_gold_df[test_gold_df['Affect Dimension'] == 'anger']
            only_fear_test = test_gold_df[test_gold_df['Affect Dimension'] == 'fear']
            only_joy_test = test_gold_df[test_gold_df['Affect Dimension'] == 'joy']
            only_sadness_test = test_gold_df[test_gold_df['Affect Dimension'] == 'sadness']

            # Classification task
            if task_name.casefold() == 'c':

                none_data_test = test_gold_df[test_gold_df['Affect Dimension'] == 'none']  # All the 11 emotions are 0:
                other_data_test = test_gold_df[test_gold_df['Affect Dimension'] == 'other']  # All the 4 emotions are 0, but not all 11:

                # Convert the emotion into 'other' for each emotion = 20226
                none_to_other_data = none_data_test.replace('none', 'other')
                anger_to_other_data = only_anger_test.replace('anger', 'other')
                fear_to_other_data = only_fear_test.replace('fear', 'other')
                joy_to_other_data = only_joy_test.replace('joy', 'other')
                sadness_to_other_data = only_sadness_test.replace('sadness', 'other')

                # Remove duplicates, keeping the corresponding Emotion X
                # Anger: 18406 Vs 1820, Fear: 18031 Vs 2195, Joy: 18511 Vs 1715, Sadness: 18295 Vs 1931
                anger_test_data = pd.concat([only_anger_test, none_to_other_data, other_data_test, fear_to_other_data, joy_to_other_data,
                     sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                fear_test_data = pd.concat([only_fear_test, none_to_other_data, other_data_test, anger_to_other_data, joy_to_other_data,
                     sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                joy_test_data = pd.concat([only_joy_test, none_to_other_data, other_data_test, anger_to_other_data, fear_to_other_data,
                     sadness_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')
                sadness_test_data = pd.concat([only_sadness_test, none_to_other_data, other_data_test, anger_to_other_data, fear_to_other_data,
                     joy_to_other_data], ignore_index=True).drop_duplicates(subset=['Tweet'], keep='first')


            # Regression task
            elif task_name.casefold() == 'r':
                # Do nothing. Reassign to emotion_data
                anger_test_data = only_anger_test  # 1002
                fear_test_data = only_fear_test  # 986
                joy_test_data = only_joy_test  # 1105
                sadness_test_data = only_sadness_test  # 975

            emotion_data_dict = {
                'anger': [anger_test_data],
                'fear': [fear_test_data],
                'joy': [joy_test_data],
                'sadness': [sadness_test_data]
            }

            with open(pickle_file_name, 'wb') as outfile:
                pickle.dump(emotion_data_dict, outfile)
            return emotion_data_dict.get(emotion)

    def split_data_into_X_and_y(emotion, dataframe, task_name):
        """
        A Method to prepare X and y data for the emotion depending upon the task.
        """
        X, y = df, df

        if task_name == 'c':
            X = dataframe.iloc[:, [0, 1, 2]]
            y = dataframe['Affect Dimension'].astype('category').cat.rename_categories({emotion: 1, 'other': 0}) # Target variable for classification task

        elif task_name == 'r':
            X = dataframe.iloc[:, [0, 1, 2]]
            y = pd.to_numeric(dataframe['Intensity Score'])  # Target variable for regression task
        return X, y
