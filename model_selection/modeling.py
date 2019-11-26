# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

from data_preparation.data_manager import Data_Manager
from data_preparation.sem_eval_task1_data import SemEval_Task1_Data
from model_selection.trans_feat_analyzer import Trans_Feat_Analyzer
from model_selection.trans_feat_rank_analyzer import Trans_Feat_Rank_Analyzer
from model_selection.vect_feat_analyzer import Vect_Feat_Analyzer
from model_selection.class_reg_all_feat import Class_Reg_All_Feat
from model_selection.dictionaries import Dictionaries
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def train_it_again(emotion, task_name, anal, k):
    """
    An user interactive method for complete training of all the models
    """
    # Select the best training model for each emotion depending upon the task name
    if (task_name.casefold() == 'c') | (task_name.casefold() == 'r'):
        # Merge data depending upon the task name
        training_data = SemEval_Task1_Data.merge_training_data(task_name)

        # Get a list of training and test data depending upon the emotion
        emotion_training_data = Data_Manager.get_train_data_for_emotion(emotion, training_data, task_name, 'train')

        # Split the training and test data into X and y, depending upon the task name
        X_train, y_train = Data_Manager.split_data_into_X_and_y(emotion, emotion_training_data, task_name)

        # Choose the estimator, depending upon the task
        estimator_dict = {}
        if task_name == 'c':
            estimator_dict = Dictionaries.classifier_dict
            if anal == 'feat':
                estimator_dict = Dictionaries.feat_anal_classifier_dict
        elif task_name == 'r':
            estimator_dict = Dictionaries.regressor_dict
            if anal == 'feat':
                estimator_dict = Dictionaries.feat_anal_regressor_dict

        # Train the model with features extracted and transformed only.
        if (anal == 'feat'):
            Trans_Feat_Analyzer.perform(emotion, X_train, y_train, task_name)
            Trans_Feat_Rank_Analyzer.perform(emotion, X_train, y_train, task_name, estimator_dict)
        # Train the model with vectorized features only.
        elif (anal == 'vect'):
            Vect_Feat_Analyzer.perform(emotion, X_train, y_train, task_name, estimator_dict, Dictionaries.vectorizer_dict)
        # Train the model with the selected number of features k depending upon emotion and task
        elif (anal == 'model'):
            Class_Reg_All_Feat.perform(emotion, X_train, y_train, task_name, k, estimator_dict, Dictionaries.vectorizer_dict)
    else:
        print("The task name is incorrect. Please restart the program and type 'c' or 'r'")


def main():
    #Default values for emotion, task_name, anal, and k
    emotion = 'anger'
    task_name = 'c'
    anal= 'model'
    k = 0

    # Get parameters from the user
    print(Dictionaries.emo_dict)
    emotion_num = int(input("Please select the emotion, by entering (only) the number: "))
    if emotion_num in (1,2,3,4):
        emotion = Dictionaries.emo_dict[emotion_num]
    else:
        print('Wrong entry. Please re-run the program and type a valid number for the emotion')
        sys.exit(1)


    task_code = input("Please choose [c] for classification task and [r] for regression task: ")
    if task_code in ('c', 'r'):
        task_name = task_code
    else:
        print('Wrong entry. Please re-run the program and type either [c] or [r].')
        sys.exit(1)


    anal_code = input("Please select [m] for model analysis, [v] for vectorizer analysis and [f] for feature analysis: ")
    if anal_code in ('m', 'v', 'f'):
        anal = Dictionaries.anal_dict[anal_code]
    else:
        print('Wrong entry. Please re-run the program and type either [m], [v] or [f].')
        sys.exit(1)

    if task_name == 'c':
        k_num = int(input("Please choose one of the k values for number of feature selection [100, 250,500,1000,1500] or [0] for selecting all the features/feat analyis/vector analysis: "))
        if k_num in (0,100, 250,500,1000,1500):
            k = k_num
        else:
            print('Wrong entry. Please re-run the program and type a valid k value')
            sys.exit(1)
    if task_name == 'r':
        # Total features after vectorization and adding transformed features = 784(anger), 986(fear), 658(joy), 742(sadness)
        k_num = int(input("Please choose one of the k values for number of feature selection [50, 100, 200, 400, 650] \nor [0] for selecting all the features: "))
        if k_num in (0, 50, 100, 200, 400, 650):
            k = k_num
        else:
            print('Wrong entry. Please re-run the program and type a valid k value')
            sys.exit(1)

    # Train using the parameters
    print('Training the model...')
    train_it_again(emotion, task_name, anal, k)

# init
if __name__ == '__main__':
    main()