from model_selection.dictionaries import Dictionaries
import pandas as pd
from pathlib import Path

class Score_Util:
    """
    A class for analyzing scores and selecting the model components of the best models
    """
    class_k_values = ['all_in', '1500', '1000', '500', '250', '100']
    reg_k_values =['all_in', '650', '400', '200', '100', '50']

    def read_csv_into_df(path_name):
        parent_dir = Path.cwd().parent
        #print(parent_dir)
        documents_dir = parent_dir.joinpath('default_results', 'score_files', path_name + '.csv')
        df = pd.read_csv(documents_dir, delimiter=",")
        return df

    def get_model_for_index(merged_df, row_index_dict, task_name):
        """
         A method to retrieve the model components using the row index of the best score
        """
        model_properties={}
        if task_name == 'classification_':
            df_original = merged_df[merged_df['training_dataset'] == 'original']
            df_resampled = merged_df[merged_df['training_dataset'] != 'original']

            #Get the best model properties of original dataset
            dataset_original = df_original.loc[row_index_dict['original'], 'training_dataset']
            estimator_original = df_original.loc[row_index_dict['original'], 'classifier']
            vectorizer_original = df_original.loc[row_index_dict['original'], 'vectorizer']
            k_value_original = df_original.loc[row_index_dict['original'], 'features_selected']
            avg_f1_original = df_original.loc[row_index_dict['original'], 'avg_cv_f1']
            f1_cv_scores_original = df_original.loc[row_index_dict['original'], 'f1_cv_scores']
            emotion_original = df_original.loc[row_index_dict['original'], 'emotion']

            # Get the best model properties of resampled dataset
            dataset_resampled= df_resampled.loc[row_index_dict['resampled'], 'training_dataset']
            estimator_resampled = df_resampled.loc[row_index_dict['resampled'], 'classifier']
            vectorizer_resampled = df_resampled.loc[row_index_dict['resampled'], 'vectorizer']
            k_value_resampled = df_resampled.loc[row_index_dict['resampled'], 'features_selected']
            avg_f1_resampled = df_resampled.loc[row_index_dict['resampled'], 'avg_cv_f1']
            f1_cv_scores_resampled = df_resampled.loc[row_index_dict['resampled'], 'f1_cv_scores']
            emotion_resampled = df_resampled.loc[row_index_dict['resampled'], 'emotion']

            model_properties['original'] = [dataset_original, estimator_original, vectorizer_original, k_value_original, avg_f1_original, f1_cv_scores_original, emotion_original]
            model_properties['resampled'] = [dataset_resampled, estimator_resampled, vectorizer_resampled, k_value_resampled, avg_f1_resampled, f1_cv_scores_resampled, emotion_resampled]
        if task_name == 'regression_':
            estimator = merged_df.loc[row_index_dict['original'], 'regressor']
            vectorizer = merged_df.loc[row_index_dict['original'], 'vectorizer']
            k_value = merged_df.loc[row_index_dict['original'], 'features_selected']
            avg_rmse = merged_df.loc[row_index_dict['original'], 'avg_cv_rmse']
            rmse_cv_scores = merged_df.loc[row_index_dict['original'], 'rmse_cv_scores']
            emotion = merged_df.loc[row_index_dict['original'], 'emotion']
            model_properties['original'] = ['original', estimator, vectorizer, k_value, avg_rmse, rmse_cv_scores,emotion]
        return model_properties

    def get_row_index_for_best_model(merged_df, score_name, task_name):
        """
        A method to find the row index of the best model depending upon the score and task
        """
        row_index_dict = {}
        if task_name == 'classification_':
            df_original = merged_df[merged_df['training_dataset'] == 'original']
            df_resampled= merged_df[merged_df['training_dataset'] != 'original']

            score_list_original = df_original[score_name].values
            score_list_resampled = df_resampled[score_name].values

            row_index_original = df_original[df_original[score_name] == max(score_list_original)].index.item()
            row_index_resampled = df_resampled[df_resampled[score_name] == max(score_list_resampled)].index.item()

            row_index_dict['original'] = row_index_original
            row_index_dict['resampled'] = row_index_resampled
        elif task_name ==  'regression_':
            score_list = merged_df[score_name].values
            row_index = merged_df[merged_df[score_name] == min(score_list)].index.item()
            row_index_dict['original'] = row_index
        return row_index_dict


    def get_merged_df_for_emotion(emotion, task_name):
        """
        A method to merge all the result dataframes depending upon the emotion
        """
        merged_df = pd.DataFrame
        if task_name == 'classification_':
            df_all_in = Score_Util.read_csv_into_df(task_name + emotion + '_feat_0').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
            df_1500 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_1500').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_1000 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_1000').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_500 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_500').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_250 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_250').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_100 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_100').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_list = [df_all_in, df_1500, df_1000, df_500, df_250, df_100]
            merged_df = pd.concat(df_list, ignore_index=True)
        elif task_name == 'regression_':
            df_all_in = Score_Util.read_csv_into_df(task_name + emotion + '_feat_0').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection
            df_650 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_650').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_400 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_400').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_200 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_200').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_100 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_100').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_50 = Score_Util.read_csv_into_df(task_name + emotion + '_feat_50').dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
            df_list = [df_all_in, df_650, df_400, df_200, df_100, df_50]
            merged_df = pd.concat(df_list, ignore_index=True, sort=True)
        return merged_df

    def select_the_best_model(task_name):
        """
        A method to select the best model depending upon the task
        """
        model_dict = {}
        for i, emotion in Dictionaries.emo_dict.items():
            merged_df = Score_Util.get_merged_df_for_emotion(emotion, task_name)
            row_index_dict={}
            # Get classification scores

            if task_name == 'classification_':
                row_index_dict = Score_Util.get_row_index_for_best_model(merged_df, 'avg_cv_f1', task_name)
                model_properties = Score_Util.get_model_for_index(merged_df, row_index_dict, task_name)
                model_dict[emotion + '_class_original'] = model_properties['original']
                model_dict[emotion + '_class_resampled'] = model_properties['resampled']
            elif task_name == 'regression_':
                row_index_dict = Score_Util.get_row_index_for_best_model(merged_df, 'avg_cv_rmse', task_name)
                model_properties = Score_Util.get_model_for_index(merged_df, row_index_dict, task_name)
                model_dict[emotion + '_reg_original'] = model_properties['original']

        return model_dict, merged_df, row_index_dict





