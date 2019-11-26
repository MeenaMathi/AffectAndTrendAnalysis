from data_preparation.sem_eval_task1_data import SemEval_Task1_Data
from data_preparation.data_manager import Data_Manager
from model_selection.dictionaries import Dictionaries
from pandas import DataFrame as df
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter
import os
import sys

"""
This class analyzes the SemEval 2018 task 1 data and the training and test data used in the project
"""
def analyze_sem_eval_task1_data():
    parent_dir = Path.cwd().parent
    documents_dir = parent_dir.joinpath('documents')

    # SEMEVAL TASK 1 - EI-reg data
    # Train
    emo_reg_anger_train_dataset = pd.read_csv(documents_dir.joinpath('El-reg-En-anger-train.txt'), delimiter="\t")
    emo_reg_fear_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-fear-train.txt'), delimiter="\t")
    emo_reg_joy_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-joy-train.txt'), delimiter="\t")
    emo_reg_sadness_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-sadness-train.txt'), delimiter="\t")

    # Dev
    emo_reg_anger_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-anger-dev.txt'), delimiter="\t")
    emo_reg_fear_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-fear-dev.txt'), delimiter="\t")
    emo_reg_joy_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-joy-dev.txt'), delimiter="\t")
    emo_reg_sadness_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-sadness-dev.txt'), delimiter="\t")

    # Test-gold
    emo_reg_anger_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-anger-test-gold.txt'),delimiter="\t")
    emo_reg_fear_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-fear-test-gold.txt'), delimiter="\t")
    emo_reg_joy_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-joy-test-gold.txt'), delimiter="\t")
    emo_reg_sadness_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-sadness-test-gold.txt'),delimiter="\t")

    # SEMEVAL TASK 1 - E-c data
    e_c_11_train_dataset = pd.read_csv(documents_dir.joinpath('2018-E-c-En-train.txt'), delimiter="\t")
    e_c_11_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-E-c-En-dev.txt'), delimiter="\t")
    e_c_11_test_dataset = pd.read_csv(documents_dir.joinpath('2018-E-c-En-test-gold.txt'), delimiter="\t")
    e_c_4_train_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-train.txt'))  # 6838
    e_c_4_dev_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-dev.txt'))
    e_c_4_test_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-test-gold.txt'))

    # SEMEVAL TASK 1 - EI-reg data
    print('SEMEVAL TASK 1 - EI-reg data : Train, Dev, Test-gold Data\n')
    print('ANGER', emo_reg_anger_train_dataset.shape[0], emo_reg_anger_dev_dataset.shape[0], emo_reg_anger_test_dataset.shape[0], sep='\t\t' )
    print('FEAR', emo_reg_fear_train_dataset.shape[0], emo_reg_fear_dev_dataset.shape[0],emo_reg_fear_test_dataset.shape[0], sep='\t\t')
    print('JOY\t', emo_reg_joy_train_dataset.shape[0], emo_reg_joy_dev_dataset.shape[0],emo_reg_joy_test_dataset.shape[0], sep='\t\t')
    print('SADNESS', emo_reg_sadness_train_dataset.shape[0], emo_reg_sadness_dev_dataset.shape[0],emo_reg_sadness_test_dataset.shape[0], sep='\t\t')

    print('\n\nSEMEVAL TASK 1 - E-c data: Train, Dev, Test-gold\n')
    print('ANGER', e_c_11_train_dataset[e_c_11_train_dataset['anger'] == 1].shape[0], e_c_11_dev_dataset[e_c_11_dev_dataset['anger'] == 1].shape[0],
          e_c_11_test_dataset[e_c_11_test_dataset['anger'] == 1].shape[0], sep='\t\t')
    print('FEAR', e_c_11_train_dataset[e_c_11_train_dataset['fear'] == 1].shape[0], e_c_11_dev_dataset[e_c_11_dev_dataset['fear'] == 1].shape[0],
          e_c_11_test_dataset[e_c_11_test_dataset['fear'] == 1].shape[0], sep='\t\t')
    print('JOY\t', e_c_11_train_dataset[e_c_11_train_dataset['joy'] == 1].shape[0], e_c_11_dev_dataset[e_c_11_dev_dataset['joy'] == 1].shape[0],
          e_c_11_test_dataset[e_c_11_test_dataset['joy'] == 1].shape[0], sep='\t\t')
    print('SADNESS', e_c_11_train_dataset[e_c_11_train_dataset['sadness'] == 1].shape[0], e_c_11_dev_dataset[e_c_11_dev_dataset['sadness'] == 1].shape[0],
          e_c_11_test_dataset[e_c_11_test_dataset['sadness'] == 1].shape[0], sep='\t\t')
    print('OTHER',e_c_4_train_dataset[e_c_4_train_dataset['Affect Dimension'] == 'other'].shape[0], e_c_4_dev_dataset[e_c_4_dev_dataset['Affect Dimension'] == 'other'].shape[0],
          e_c_4_test_dataset[e_c_4_test_dataset['Affect Dimension'] == 'other'].shape[0], sep='\t\t')
    print('NONE', e_c_4_train_dataset[e_c_4_train_dataset['Affect Dimension'] == 'none'].shape[0], e_c_4_dev_dataset[e_c_4_dev_dataset['Affect Dimension'] == 'none'].shape[0],
          e_c_4_test_dataset[e_c_4_test_dataset['Affect Dimension'] == 'none'].shape[0], sep='\t\t' )

def get_train_and_test_data_for_task(taskname):
    dataset_train, dataset_gold = df, df
    if taskname == 'c':
        # Classification task - Training data
        dataset_train = SemEval_Task1_Data.merge_training_data('c')
        print('Classification SemEval task1: train + dev data of EI-reg and E-c data')
        print(dataset_train.shape[0])  # 13542
        print('Classification SemEval task1: train + dev - anger, fear, joy, sadness, none, other')
        print(dataset_train[dataset_train['Affect Dimension'] == 'anger'].shape[0])  # 3542
        print(dataset_train[dataset_train['Affect Dimension'] == ('fear')].shape[0])  # 2879
        print(dataset_train[dataset_train['Affect Dimension'] == ('joy')].shape[0])  # 3228
        print(dataset_train[dataset_train['Affect Dimension'] == ('sadness')].shape[0])  # 3156
        print(dataset_train[dataset_train['Affect Dimension'] == ('none')].shape[0])  # 218
        print(dataset_train[dataset_train['Affect Dimension'] == ('other')].shape[0])  # 519

        # Classification task - Test gold data
        dataset_gold = SemEval_Task1_Data.merge_test_data('c')
        print('Classification SemEval task1: Complete test gold data of EI-reg and E-c data')
        print(dataset_gold.shape[0])  # 73475
        print('Classification SemEval task1: test gold - anger, fear, joy, sadness, none, other')
        print(dataset_gold[dataset_gold['Affect Dimension'] == 'anger'].shape[0])  # 18406
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('fear')].shape[0])  # 18031
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('joy')].shape[0])  # 18511
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('sadness')].shape[0])  # 18295
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('none')].shape[0])  # 75
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('other')].shape[0])  # 157

    elif taskname == 'r':
        # Regression task - Training data
        dataset_train = SemEval_Task1_Data.merge_training_data('r')
        print('Regression SemEval task1: train + dev data of E-Reg data - Training data used for regression in this project')
        print(dataset_train.shape[0])  #8566
        print('Regression SemEval task1: train + dev - anger, fear, joy, sadness - Training data for each emotion in this project')
        print(dataset_train[dataset_train['Affect Dimension'] == 'anger'].shape[0])  #2089
        print(dataset_train[dataset_train['Affect Dimension'] == ('fear')].shape[0])  #2641
        print(dataset_train[dataset_train['Affect Dimension'] == ('joy')].shape[0])  #1906
        print(dataset_train[dataset_train['Affect Dimension'] == ('sadness')].shape[0])  #1930

        # Regression task - Test gold data
        dataset_gold = SemEval_Task1_Data.merge_test_data('r')
        print()
        print('Regression SemEval task1: Complete test gold data ')
        print(dataset_gold.shape[0])  #4068
        print('Regression SemEval task1: test gold - anger, fear, joy, sadness - Complete test data for each emotion')
        print(dataset_gold[dataset_gold['Affect Dimension'] == 'anger'].shape[0])  #1002
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('fear')].shape[0])  #986
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('joy')].shape[0])  #1105
        print(dataset_gold[dataset_gold['Affect Dimension'] == ('sadness')].shape[0]) #975
        print('Regression SemEval task1: test gold - anger, fear, joy, sadness - Test data for each emotion used in this project')
        print(int(dataset_train[dataset_train['Affect Dimension'] == 'anger'].shape[0] * 0.25))  # 2089 * 20/80
        print(int(dataset_train[dataset_train['Affect Dimension'] == ('fear')].shape[0] * 0.25)) # 2641 * 20/80
        print(int(dataset_train[dataset_train['Affect Dimension'] == ('joy')].shape[0] * 0.25))  # 1906 * 20/80
        print(int(dataset_train[dataset_train['Affect Dimension'] == ('sadness')].shape[0] * 0.25))  # 1930 * 20/80

    return dataset_train, dataset_gold

def check_for_duplicates(dataset_train, dataset_gold):
    # TEST 1 - Check for Discrete data in training data =>
    print('Unique tweets in training data (CodaLab train + dev)')
    clean_train = dataset_train.drop_duplicates(subset=['ID', 'Tweet'], keep='first')
    print(clean_train.shape[0]) #7801

    # TEST 2 - Check for Discrete data in test data =>
    print('Unique tweets in test data (CodaLab test gold)')
    clean_gold = dataset_gold.drop_duplicates(subset=['ID', 'Tweet'], keep='first')
    print(clean_gold.shape[0]) #20226


def analyze_class_emo_train_and_test_data(anger_data, fear_data, joy_data, sadness_data):
    print()
    print('ANGER - Training data and test data used in this project')
    print('Total\tAnger\tOther')
    print(anger_data[0].shape[0],anger_data[0]['Affect Dimension'].value_counts()[1], anger_data[0]['Affect Dimension'].value_counts()[0], sep='\t' )
    print(int(anger_data[0].shape[0] * 0.25)-1, int(anger_data[0]['Affect Dimension'].value_counts()[1] * 0.25),int(anger_data[0]['Affect Dimension'].value_counts()[0] * 0.25), sep='\t')
    print()

    print('FEAR - Training data and test data used in this project')
    print('Total\tFear\tOther')
    print(fear_data[0].shape[0], fear_data[0]['Affect Dimension'].value_counts()[1],fear_data[0]['Affect Dimension'].value_counts()[0], sep='\t')
    print(int(fear_data[0].shape[0]*0.25)-1, int(fear_data[0]['Affect Dimension'].value_counts()[1] *0.25), int(fear_data[0]['Affect Dimension'].value_counts()[0]*0.25), sep='\t')
    print()

    print('JOY - Training data and test data used in this project')
    print('Total\tJoy\tOther')
    print(joy_data[0].shape[0], joy_data[0]['Affect Dimension'].value_counts()[1],joy_data[0]['Affect Dimension'].value_counts()[0], sep='\t')
    print(int(joy_data[0].shape[0]*0.25),int(joy_data[0]['Affect Dimension'].value_counts()[1]*0.25),int(joy_data[0]['Affect Dimension'].value_counts()[0]*0.25), sep='\t')
    print()

    print('SADNESS - Training data and test data used in this project')
    print('Total\tSadness\tOther')
    print(sadness_data[0].shape[0], sadness_data[0]['Affect Dimension'].value_counts()[1],sadness_data[0]['Affect Dimension'].value_counts()[0], sep='\t')
    print(int(sadness_data[0].shape[0]*0.25), int(sadness_data[0]['Affect Dimension'].value_counts()[1]*0.25), int(sadness_data[0]['Affect Dimension'].value_counts()[0]*0.25), sep='\t')
    print()


def analyze_resampled_class_train_data():
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')

    for i, emotion in Dictionaries.emo_dict.items():
        for vect_name, vectorizer in Dictionaries.vectorizer_dict.items():
            print('\n\nResampled data  - EMOTION: ' + emotion + ', VECTORIZER: '+ vect_name)
            preprocess_train_df, feat_transformed_train_df = df, df
            # Fit transform the vectorizer with the corresponding preprocessed training data
            if os.path.exists(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl')):
                preprocess_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl'))
                train_vect = vectorizer.fit_transform(preprocess_train_df['preprocessed_text'].values)
                print(emotion + ' vectorized features: ', train_vect.shape)
                train_vect_df = pd.DataFrame(train_vect.toarray(), columns=vectorizer.get_feature_names())
            if os.path.exists(pickle_dir.joinpath(emotion + '_c_train_feat_transform_df.pkl')):
                feat_transformed_train_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_c_train_feat_transform_df.pkl'))
                print(emotion + ' transformed features: ', feat_transformed_train_df.shape)
            else:
                # If the file doesnt exist, exit the program with instructions
                print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py')
                sys.exit(1)
            features_df = pd.concat([train_vect_df, feat_transformed_train_df], axis=1)
            print(emotion + ' merged features: ', features_df.shape)

            #Resample the training data using SMOTE, Tomek links and SMOTETomek
            smote_X_train, smote_y_train = SMOTE(random_state=42, sampling_strategy='minority', n_jobs=-1).fit_resample(features_df, preprocess_train_df['Affect Dimension'])
            tomek_X_train, tomek_y_train = TomekLinks(random_state=42, sampling_strategy='majority',n_jobs=-1).fit_resample(features_df, preprocess_train_df['Affect Dimension'])
            smotetomek_X_train, smotetomek_y_train = SMOTETomek(random_state=42).fit_resample(features_df, preprocess_train_df['Affect Dimension'])
            print('Data in SMOTE, Tomek links, SMOTETomek')
            print(smote_X_train.shape[0], tomek_X_train.shape[0], smotetomek_X_train.shape[0])
            print(Counter(smote_y_train), Counter(tomek_y_train), Counter(smotetomek_y_train))


# Main method
def main():
    #Analyze SEMEVAL Task 1: AIT
    analyze_sem_eval_task1_data()

    # Classification task
    print('\n\nCLASSIFICATION TASK')
    class_train_dev_data, class_gold_data = get_train_and_test_data_for_task('c')

    #print unique tweets
    check_for_duplicates(class_train_dev_data, class_gold_data)

    #Get training and test data for analysis
    anger_training_data = Data_Manager.get_train_data_for_emotion('anger', class_train_dev_data, 'c', 'train')
    anger_test_data = Data_Manager.get_test_data_for_emotion('anger', class_gold_data, 'c', 'test')
    fear_training_data = Data_Manager.get_train_data_for_emotion('fear', class_train_dev_data,'c','train')
    fear_test_data = Data_Manager.get_test_data_for_emotion('fear', class_gold_data,'c','test')
    joy_training_data = Data_Manager.get_train_data_for_emotion('joy', class_train_dev_data,  'c', 'train')
    joy_test_data = Data_Manager.get_test_data_for_emotion('joy', class_gold_data, 'c', 'test')
    sadness_training_data = Data_Manager.get_train_data_for_emotion('sadness', class_train_dev_data,  'c', 'train')
    sadness_test_data = Data_Manager.get_test_data_for_emotion('sadness', class_gold_data, 'c', 'test')
    # Analyze data for each emotion
    analyze_class_emo_train_and_test_data([anger_training_data, anger_test_data], [fear_training_data, fear_test_data], [joy_training_data, joy_test_data], [sadness_training_data, sadness_test_data])
    # Analyze resampled training data
    analyze_resampled_class_train_data()


    # Regression task
    print('\n\nREGRESSION TASK')
    reg_train_dev_data, reg_gold_data = get_train_and_test_data_for_task('r')


# init
if __name__ == '__main__':
    main()