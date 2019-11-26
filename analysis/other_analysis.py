from analysis.data_analysis import *
import pandas as pd
from model_selection.dictionaries import Dictionaries
from model_selection.score_util import Score_Util

"""
This class analyzes various results obtained in the training phase like
a) Number of vectorized features obtained using different vectorizers in classification and regression task
b) Best models selected in classification and regression task
c) Cross validation average F1 scores obtained by the classification best models using original and resampled dataset
d) Average F1 scores obtained by the best classification and regression models in model assessment phase using test data
"""

def analyze_vectorized_features(task_name):
    parent_dir = Path.cwd().parent
    pickle_dir = parent_dir.joinpath('default_results', 'pickle_files_feat_eng')

    preprocessed_df = {}
    #Get preprocess data for each emotion depending upon the task_name
    for i, emotion in Dictionaries.emo_dict.items():
        if os.path.exists(pickle_dir.joinpath(emotion + '_c_train_preprocess_df.pkl')):
            emo_preprocess_df = pd.read_pickle(pickle_dir.joinpath(emotion + '_' + task_name+'_train_preprocess_df.pkl'))
            preprocessed_df[emotion] = emo_preprocess_df
        else:
            # If the file doesnt exist, exit the program with instructions
            print('\nRequired files does not exist.\n\n Please, train the models first by running > Modelling.py and add the files created in \'default_results\' folder')
            sys.exit(1)

    task_title = 'Classification' # Default title
    if task_name == 'r':
        task_title = 'Regression'

    print('\nVectorized features in ', task_title)
    for emotion, df in preprocessed_df.items():
        for vect_name, vectorizer in Dictionaries.vectorizer_dict.items():
            vect_feat = vectorizer.fit_transform(df['preprocessed_text'].values)
            vect_df = pd.DataFrame(vect_feat.toarray(), columns=vectorizer.get_feature_names())
            print(emotion.capitalize(), ' - ', vect_name, ': ', vect_df.shape[1], sep='')

def get_best_models(task_name):
    parent_dir = Path.cwd().parent
    result_file = parent_dir.joinpath('default_results', 'score_files', 'final_best_models.csv' )
    result_dataframe = pd.DataFrame
    if os.path.exists(result_file):
        result_dataframe = pd.read_csv(result_file)
    else:
        # If the file doesnt exist, exit the program with instructions
        print('\nRequired files does not exist.\n\n Please, train and assess the models first by running > modeling.py, model_assessment.py and add the files created in \'default_results\' folder')
        sys.exit(1)

    #print(result_dataframe)
    anger_results = result_dataframe['anger_' + task_name].tolist()
    fear_results = result_dataframe['fear_' + task_name].tolist()
    joy_results = result_dataframe['joy_' + task_name].tolist()
    sadness_results = result_dataframe['sadness_' + task_name].tolist()

    task_title = '\nClassification Best models and cv scores' # Default title
    avg_score_name = 'average cv F1 score'
    cv_score_name = 'CV F1 scores'

    if task_name == 'r':
        task_title = '\nRegression Best models and cv scores'
        avg_score_name = 'average cv RMSE score'
        cv_score_name = 'CV RMSE scores'

    print(task_title)
    print('[dataset, estimator, vectorizer, #selected features, '+ avg_score_name + ', ' + cv_score_name + ']')
    print(anger_results, fear_results, joy_results, sadness_results, sep='\n')

def get_scores_of_both_class_models():
    parent_dir = Path.cwd().parent
    result_file = parent_dir.joinpath('default_results', 'score_files', 'best_class_both_model_scores.pkl' )
    results = pd.DataFrame
    #Print the model properties
    print('Classification 2 best models properties using original and resampled dataset')
    dict, merged_df, row_index_dict = Score_Util.select_the_best_model('classification_')
    print(dict)
    #Print the classification report
    if os.path.exists(result_file):
        results = pd.read_pickle(result_file)
        print('\n\nClassification report of best classification model using original and resampled dataset')
        for emo_model, results in results.items():
            print(emo_model)
            print(results)
            print()
    else:
        # If the file doesnt exist, exit the program with instructions
        print('\nRequired files does not exist.\n\n Please, train and assess the models first by running > modeling.py, model_assessment.py and add the files created in \'default_results\' folder')
        sys.exit(1)

def get_test_scores_for_best_model(task_name):
    parent_dir = Path.cwd().parent
    result_folder = parent_dir.joinpath('default_results', 'score_files')
    result_file = result_folder.joinpath(task_name + '_results_testdata.csv')
    result_dataframe = pd.DataFrame
    if os.path.exists(result_file):
        result_dataframe = pd.read_csv(result_file).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)
        print('Assessment results of best ' + task_name + ' model using test data')
        result_list =[]
        for i, rows in result_dataframe.iterrows():
            result_list.append(list(rows))
        print(result_list)
    else:
        # If the file doesnt exist, exit the program with instructions
        print(
            '\nRequired files does not exist.\n\n Please, train and assess the models first by running > modeling.py, model_assessment.py and add the files created in \'default_results\' folder')
        sys.exit(1)



def main():
    analyze_vectorized_features('c')
    analyze_vectorized_features('r')
    get_best_models('c')
    get_best_models('r')
    get_scores_of_both_class_models()
    get_test_scores_for_best_model('classification')
    get_test_scores_for_best_model('regression')


# init
if __name__ == '__main__':
    main()