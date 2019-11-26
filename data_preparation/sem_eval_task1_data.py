import pandas as pd
import os
from pathlib import Path




class SemEval_Task1_Data:
    """
    A class to prepare data from the SemEval 2018 task 1 EI-reg and E-c data
    Reference: Mohammad, Saif, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko.
    "Semeval-2018 task 1: Affect in tweets." In Proceedings of The 12th International Workshop on Semantic Evaluation, pp. 1-17. 2018.
    """

    def convert_emo11_to_emo4_with_labels(file_name):
        """
        A method to convert the format of E-c data similar to EI-reg data
        """
        dataset = pd.read_csv(file_name, delimiter="\t")

        #Create a new data frame similar to Emo-reg-dataset
        col_names = ['ID', 'Tweet', 'Affect Dimension', 'Intensity Score']
        new_dataset = pd.DataFrame(columns=col_names)

        #Handle missing indices by reindexing
        dataset = dataset.reindex(range(0, len(dataset) + 1))

        for i in range(len(dataset)):
            # (Only)if all the emotions are 0, the tweet has no emotion[none]
            if ((dataset.at[i,'anger'] == 0) & (dataset.at[i,'fear'] == 0) & (dataset.at[i,'joy'] == 0) &
                (dataset.at[i, 'sadness'] == 0) & (dataset.at[i,'anticipation'] == 0) &  (dataset.at[i,'love'] == 0) &
                (dataset.at[i, 'optimism'] == 0) & (dataset.at[i,'pessimism'] == 0) & (dataset.at[i,'trust'] == 0)
                & (dataset.at[i, 'surprise'] == 0) & (dataset.at[i,'disgust'] == 0)):
                new_data = {'ID': dataset.at[i, 'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'none', 'Intensity Score': 'none'}
                new_dataset.loc[len(new_dataset)] = new_data

            # If the 4 emotions are 0 and the tweet has one among the other emotions, the tweet is said to have [other] emotion
            if ((dataset.at[i, 'anger'] == 0) & (dataset.at[i, 'fear'] == 0) & (dataset.at[i, 'joy'] == 0) & (dataset.at[i, 'sadness'] == 0)):
                if ((dataset.at[i,'anticipation'] == 1) |  (dataset.at[i,'love'] == 1) | (dataset.at[i, 'optimism'] == 1)
                        | (dataset.at[i,'pessimism'] == 1) | (dataset.at[i,'trust'] == 1) | (dataset.at[i, 'surprise'] == 1) | (dataset.at[i,'disgust'] == 1)):
                    new_data = {'ID': dataset.at[i, 'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'other','Intensity Score': 'none'}
                    new_dataset.loc[len(new_dataset)] = new_data
            # Tweets with anger emotion
            if (dataset.at[i,'anger'] == 1):
                new_data = {'ID': dataset.at[i, 'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'anger','Intensity Score': 'none'}
                new_dataset.loc[len(new_dataset)] = new_data
            # Tweets with fear emotion
            if (dataset.at[i, 'fear'] == 1):
                new_data = {'ID': dataset.at[i, 'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'fear','Intensity Score': 'none'}
                new_dataset.loc[len(new_dataset)] = new_data
            # Tweets with joy emotion
            if (dataset.at[i, 'joy'] == 1):
                new_data = {'ID': dataset.at[i, 'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'joy','Intensity Score': 'none'}
                new_dataset.loc[len(new_dataset)] = new_data
            # Tweets with sadness emotion
            if (dataset.at[i, 'sadness'] == 1):
                new_data = {'ID': dataset.at[i,'ID'], 'Tweet': dataset.at[i, 'Tweet'], 'Affect Dimension': 'sadness', 'Intensity Score': 'none'}
                new_dataset.loc[len(new_dataset)] = new_data

        return(new_dataset)


    def merge_training_data(task_name):
        """
        A method to prepare training data depending upon the task, by merging EI-reg and E-c train+dev data
        """
        parent_dir = Path.cwd().parent
        documents_dir = parent_dir.joinpath('documents')

        #Read the files
        # Emo-reg files
        Emo_reg_anger_train_dataset = pd.read_csv(documents_dir.joinpath('El-reg-En-anger-train.txt'), delimiter="\t")
        Emo_reg_fear_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-fear-train.txt'), delimiter="\t")
        Emo_reg_joy_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-joy-train.txt'), delimiter="\t")
        Emo_reg_sadness_train_dataset = pd.read_csv(documents_dir.joinpath('EI-reg-En-sadness-train.txt'), delimiter="\t")
        Emo_reg_anger_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-anger-dev.txt'), delimiter="\t")
        Emo_reg_fear_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-fear-dev.txt'), delimiter="\t")
        Emo_reg_joy_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-joy-dev.txt'), delimiter="\t")
        Emo_reg_sadness_dev_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-sadness-dev.txt'), delimiter="\t")

        # E-c-files: Modify the format similar to Emo-reg files
        e_c_4_train_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-train.txt'))  # 6838
        e_c_4_dev_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-dev.txt'))

        pickle_dir = parent_dir.joinpath('new_results','pickle_files_feat_eng')
        # Create a directory for storing pickle files, if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        # Create the final merged file without duplicate data for emotion_classification task
        if (task_name == 'c'):
            input_file_name = pickle_dir.joinpath('class_training_data.pkl')
            # Check the existence of the file and if the file is not empty
            if os.path.exists(input_file_name) and os.path.getsize(input_file_name) > 0:
                emo_class_final = pd.read_pickle(input_file_name)
                return emo_class_final
            else:
                emo_class_dataset_list = [Emo_reg_anger_train_dataset, Emo_reg_fear_train_dataset,
                                          Emo_reg_joy_train_dataset, Emo_reg_sadness_train_dataset,
                                          Emo_reg_anger_dev_dataset, Emo_reg_fear_dev_dataset,
                                          Emo_reg_joy_dev_dataset, Emo_reg_sadness_dev_dataset,
                                          e_c_4_train_dataset, e_c_4_dev_dataset]
                emo_class_concat = pd.concat(emo_class_dataset_list)
                emo_class_final = emo_class_concat.drop_duplicates(subset=['ID', 'Tweet', 'Affect Dimension'])
                emo_class_final.to_pickle(input_file_name)
                return emo_class_final

        # Create the final merged file without duplicate data for emotion_regression task
        elif (task_name == 'r'):
            input_file_name = pickle_dir.joinpath('reg_training_data.pkl')
            # Check the existence of the file and if the file is not empty
            if os.path.exists(input_file_name) and os.path.getsize(input_file_name) > 0:
                emo_reg_final = pd.read_pickle(input_file_name)
                return emo_reg_final
            else:
                emo_reg_dataset_list = [Emo_reg_anger_train_dataset, Emo_reg_fear_train_dataset,
                                          Emo_reg_joy_train_dataset, Emo_reg_sadness_train_dataset,
                                          Emo_reg_anger_dev_dataset, Emo_reg_fear_dev_dataset,
                                          Emo_reg_joy_dev_dataset, Emo_reg_sadness_dev_dataset]
                emo_reg_final = pd.concat(emo_reg_dataset_list).drop_duplicates(subset=['ID', 'Tweet', 'Affect Dimension', 'Intensity Score'])
                emo_reg_final.to_pickle(input_file_name)
                return emo_reg_final


    def merge_test_data(task_name):
        """
        A method to prepare test data depending upon the task, by merging EI-reg and E-c test-gold data
        """
        parent_dir = Path.cwd().parent
        documents_dir = parent_dir.joinpath('documents')
        #Read the text files
        # Emo-reg files
        Emo_reg_anger_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-anger-test-gold.txt'),delimiter="\t")
        Emo_reg_fear_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-fear-test-gold.txt'), delimiter="\t")
        Emo_reg_joy_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-joy-test-gold.txt'), delimiter="\t")
        Emo_reg_sadness_test_dataset = pd.read_csv(documents_dir.joinpath('2018-EI-reg-En-sadness-test-gold.txt'), delimiter="\t")

        # E-c-files: Modify the format similar to Emo-reg files
        e_c_4_test_dataset = SemEval_Task1_Data.convert_emo11_to_emo4_with_labels(documents_dir.joinpath('2018-E-c-En-test-gold.txt'))

        pickle_dir = parent_dir.joinpath('new_results','pickle_files_feat_eng')
        # Create a directory for storing pickle files, if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        # Create the final merged file without duplicate data for emotion_classification task
        if (task_name == 'c'):
            input_file_name = pickle_dir.joinpath('class_test_data.pkl')
            # Check the existence of the file and if the file is not empty
            if os.path.exists(input_file_name) and os.path.getsize(input_file_name) > 0:
                emo_class_final = pd.read_pickle(input_file_name)
                return emo_class_final
            else:
                emo_class_dataset_list = [Emo_reg_anger_test_dataset, Emo_reg_fear_test_dataset,
                                          Emo_reg_joy_test_dataset, Emo_reg_sadness_test_dataset,
                                          e_c_4_test_dataset]
                emo_class_concat = pd.concat(emo_class_dataset_list)
                emo_class_final = emo_class_concat.drop_duplicates(subset=['ID', 'Tweet', 'Affect Dimension'])
                emo_class_final.to_pickle(input_file_name)
                return emo_class_final

        # Create the final merged file without duplicate data for emotion_regression task
        elif (task_name == 'r'):
            input_file_name = pickle_dir.joinpath('reg_test_data.pkl')
            # Check the existence of the file and if the file is not empty
            if os.path.exists(input_file_name) and os.path.getsize(input_file_name) > 0:
                emo_reg_final = pd.read_pickle(input_file_name)
                return emo_reg_final
            else:
                emo_reg_dataset_list = [Emo_reg_anger_test_dataset, Emo_reg_fear_test_dataset,
                                          Emo_reg_joy_test_dataset, Emo_reg_sadness_test_dataset]
                emo_reg_df = pd.concat(emo_reg_dataset_list).drop_duplicates(subset=['ID', 'Tweet', 'Affect Dimension', 'Intensity Score'])
                emo_reg_final = emo_reg_df[emo_reg_df['Intensity Score'] != 0.000]
                emo_reg_final.to_pickle(input_file_name)
                return emo_reg_final

