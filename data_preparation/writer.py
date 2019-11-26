from model_selection.dictionaries import Dictionaries
import pandas as pd
import numpy as np
from statistics import mean
import os
from pathlib import Path
import pickle as pickle

class Writer:
    """
    This class writes results in the corresponding files
    """

    parent_dir = Path.cwd().parent
    scores_dir = parent_dir.joinpath('new_results', 'score_files')
    results_dir = parent_dir.joinpath('new_results', 'prediction_results')
    user_data_dir = parent_dir.joinpath('new_results', 'pickle_files_user_data')

    def check_for_directory(output_folder_or_file_name):
        # Create a directory, if it doesn't exist
        if not os.path.exists(output_folder_or_file_name):
            os.makedirs(output_folder_or_file_name)

    def write_class_vect_anal_results_in_file(emotion, resampler_name, classifier_name, vect_name, scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'training_dataset', 'classifier', 'vectorizer',
                     'avg_fit_time', 'avg_score_time',
                     'f1_cv_scores', 'avg_cv_f1']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'training_dataset': resampler_name,
                    'classifier': classifier_name,
                    'vectorizer': vect_name,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'f1_cv_scores': scores['test_score'],
                    'avg_cv_f1': mean(scores['test_score'])
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('classification_' + emotion + '_vect_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_reg_vect_anal_results_in_file(emotion, regressor_name, vect_name, scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'regressor', 'vectorizer', 'avg_fit_time', 'avg_score_time', 'rmse_cv_scores','avg_cv_rmse']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'regressor': regressor_name,
                    'vectorizer': vect_name,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'rmse_cv_scores': np.sqrt(-scores['test_score']),
                    'avg_cv_rmse': mean(np.sqrt(-scores['test_score']))
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('regression_' + emotion + '_vect_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_class_feat_anal_results_in_file(emotion, resampler_name, classifier_name, features_selected, scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'training_dataset', 'classifier', 'features_selected','avg_fit_time', 'avg_score_time',
                     'f1_cv_scores', 'avg_cv_f1']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'training_dataset': resampler_name,
                    'classifier': classifier_name,
                    'features_selected': features_selected,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'f1_cv_scores': scores['test_score'],
                    'avg_cv_f1': mean(scores['test_score'])
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('classification_' + emotion + '_feat_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_reg_feat_anal_results_in_file(emotion, regressor_name, features_selected, scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'regressor', 'features_selected', 'avg_fit_time', 'avg_score_time', 'rmse_cv_scores','avg_cv_rmse']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'regressor': regressor_name,
                    'features_selected': features_selected,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'rmse_cv_scores': np.sqrt(-scores['test_score']),
                    'avg_cv_rmse': mean(np.sqrt(-scores['test_score']))
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('regression_' + emotion + '_feat_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_class_feat_rank_anal_results_in_file(emotion, resampler_name, estimator_name, selected_features, feature_mask, feature_rank, cv_feature_scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'training_dataset', 'estimator','n_features', 'support', 'ranking', 'grid_f1_scores']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'training_dataset': resampler_name,
                    'estimator': estimator_name,
                    'n_features': selected_features,
                    'support': feature_mask,
                    'ranking': feature_rank,
                    'grid_f1_scores': cv_feature_scores
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('classification' + '_' + emotion + '_feat_rank_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_reg_feat_rank_anal_results_in_file(emotion, estimator_name, selected_features, feature_mask, feature_rank, cv_feature_scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'regressor','n_features', 'support', 'ranking', 'grid_rmse_scores']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'regressor': estimator_name,
                    'n_features': selected_features,
                    'support': feature_mask,
                    'ranking': feature_rank,
                    'grid_rmse_scores': cv_feature_scores
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('regression' + '_' + emotion + '_feat_rank_anal.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)

    def write_class_model_anal_results_in_file(emotion, resampler_name, classifier_name, vect_name, k, scores):
        #parent_dir = Path.cwd().parent
        #scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        #Create a dataframe to store the result scores
        col_names = ['emotion', 'training_dataset', 'classifier', 'vectorizer', 'features_selected',
                     'avg_fit_time', 'avg_score_time',
                     'f1_cv_scores', 'avg_cv_f1']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'training_dataset': resampler_name,
                    'classifier': classifier_name,
                    'vectorizer': vect_name,
                    'features_selected': k,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'f1_cv_scores': scores['test_score'],
                    'avg_cv_f1': mean(scores['test_score'])
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('classification_' + emotion + '_feat_' + str(k) + '.csv'), mode='a',encoding='utf-8')
        print(results_dataframe)

    def write_reg_model_anal_results_in_file(emotion, regressor_name, vect_name, k, scores):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'regressor', 'vectorizer', 'features_selected', 'avg_fit_time', 'avg_score_time', 'rmse_cv_scores','avg_cv_rmse']
        results_dataframe = pd.DataFrame(columns=col_names)
        print(sorted(scores.keys()))

        new_data = {'emotion': emotion,
                    'regressor': regressor_name,
                    'vectorizer': vect_name,
                    'features_selected': k,
                    'avg_fit_time': mean(scores['fit_time']),
                    'avg_score_time': mean(scores['score_time']),
                    'rmse_cv_scores': np.sqrt(-scores['test_score']),
                    'avg_cv_rmse': mean(np.sqrt(-scores['test_score']))
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        results_dataframe.to_csv(Writer.scores_dir.joinpath('regression_' + emotion + '_feat_'+ str(k) + '.csv'), mode='a', encoding='utf-8')
        print(results_dataframe)


    def write_user_test_data_to_predict(tweets_dict, keyword, date):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.user_data_dir)
        input_file = Writer.user_data_dir.joinpath(keyword + str(date) + '.pkl')

        # Create a new data frame similar to Emo-reg-dataset
        col_names = ['ID', 'Tweet', 'Affect Dimension', 'Intensity Score', 'Date', 'Number of tweets']
        test_dataset = pd.DataFrame(columns=col_names)
        num = 0

        for tweets, list in tweets_dict.items():
            id = '2019-' + keyword + '-' + str(num)
            new_data = {'ID': id, 'Tweet': tweets, 'Affect Dimension': None, 'Intensity Score': None, 'Date': list[0], 'Number of tweets': list[2]}
            test_dataset.loc[len(test_dataset)] = new_data
            num += 1
        print(test_dataset.Tweet)
        test_dataset.to_pickle(input_file)

    def write_class_results_for_test_data(emotion, best_model_prop_list, prec, recall, accuracy, f1, avg_prec, avg_recall, avg_f1):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'training_dataset', 'classifier', 'vectorizer', 'features_selected',
                     'test_prec', 'test_recall', 'test_acc', 'test_f1',
                     'test_avg_prec', 'test_avg_recall', 'test_avg_f1']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'training_dataset': best_model_prop_list[0],
                    'classifier': best_model_prop_list[1],
                    'vectorizer': best_model_prop_list[2],
                    'features_selected': best_model_prop_list[3],
                    'test_prec': prec,
                    'test_recall': recall,
                    'test_acc': accuracy,
                    'test_f1': f1,
                    'test_avg_prec': avg_prec,
                    'test_avg_recall': avg_recall,
                    'test_avg_f1': avg_f1
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        print(results_dataframe)
        results_dataframe.to_csv(Writer.scores_dir.joinpath('classification_results_testdata.csv'), mode='a',encoding='utf-8')

    def write_reg_results_for_test_data(emotion, best_model_prop_list, mae, rmse):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)

        # Create a dataframe to store the result scores
        col_names = ['emotion', 'regressor', 'vectorizer', 'features_selected', 'test_mae', 'test_rmse']
        results_dataframe = pd.DataFrame(columns=col_names)

        new_data = {'emotion': emotion,
                    'regressor': best_model_prop_list[2],
                    'vectorizer': best_model_prop_list[3],
                    'features_selected': best_model_prop_list[0],
                    'test_mae': mae,
                    'test_rmse': rmse
                    }
        results_dataframe.loc[len(results_dataframe)] = new_data

        print(results_dataframe)
        results_dataframe.to_csv(Writer.scores_dir.joinpath('regression_results_testdata.csv'), mode='a',encoding='utf-8')


    def write_the_classification_results(user_data_with_affects):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.results_dir)
        result_file = Writer.results_dir.joinpath('class_results.csv')
        user_data_with_affects.to_csv(result_file, index=None, header=True)

    def write_the_final_results(final_results, user_name):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.results_dir)
        result_file = Writer.results_dir.joinpath('final_results_' + user_name + '.csv')
        final_results.to_csv(result_file, index=None, header=True)

    def write_the_final_predicted_results(final_results, user_name, date):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.results_dir)
        result_file = Writer.results_dir.joinpath('final_results_' + user_name + '_' + date + '.csv')
        final_results.to_csv(result_file, index=None, header=True)

    def write_the_best_model_properties(class_best_model_dict, reg_best_model_dict):
        # parent_dir = Path.cwd().parent
        # scores_dir = parent_dir.joinpath('score_files')
        Writer.check_for_directory(Writer.scores_dir)
        result_file = Writer.scores_dir.joinpath('final_best_models.csv')

        best_model_dict = {}
        for i, emotion in Dictionaries.emo_dict.items():
            best_model_dict[emotion + '_c'] = class_best_model_dict[emotion]
            best_model_dict[emotion + '_r'] = reg_best_model_dict[emotion + '_reg_original']

        best_model_df = pd.DataFrame.from_dict(best_model_dict)
        print(best_model_df)
        best_model_df.to_csv(result_file, index=None, header=True)



