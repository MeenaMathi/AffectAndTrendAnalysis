import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from model_selection.dictionaries import Dictionaries
"""
This class analyzes the relevancy of the 14 transformed features based on the selection of features using RFECV feature selection method
The results are used for ranking the features (individually for classification and regression task) based on the number of times the features get selected by different models
"""
def rank_feat_for_model(task_name, score_name):

    parent_dir = Path.cwd().parent
    score_file = parent_dir.joinpath('default_results', 'score_files')

    title = 'Classification - Transformed features' # default title
    if task_name == 'regression_':
        title = 'Regression - Transformed features'

    fig = plt.figure()
    fig.suptitle(title)
    feature_name = Dictionaries.trans_feature_list

    for i, emotion in Dictionaries.emo_dict.items():
        df_all_in_fect = pd.read_csv(score_file.joinpath(task_name + emotion + '_feat_rank_anal.csv')).dropna().drop(columns=['Unnamed: 0']).reset_index(drop=True)  # all features without feature selection

        rank_lists = df_all_in_fect[score_name].tolist()
        print(rank_lists)

        # Remove other ranks except 1, because rank 1 indicates that the feature is selected based on feature score
        new_rank_list = []
        for list in rank_lists:
            list = list.replace('[','').replace(']','').split(' ')
            list[:] = filter(None, list)
            list[:] = [1 if int(x)==1 else 0 for x in list]
            print(list)
            new_rank_list.append(list)

        number_of_hits = [sum(x) for x in zip(*new_rank_list)]

        print(feature_name)
        print('Number of hits: ', number_of_hits)

        ax = plt.subplot(2, 2, i)
        ax.bar(feature_name, number_of_hits)
        plt.title(emotion.capitalize())
        if i in (3, 4):
            plt.xlabel('Feature names', fontsize = 14)
            plt.xticks(range(len(feature_name)), feature_name, fontsize=12, rotation=90)
        else:
            plt.gca().get_xaxis().set_visible(False)
        plt.ylabel('Number of hits', fontsize = 16)
        plt.yticks([1,2,3,4,5,6,7,8,9,10,11,12,13])
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.3, wspace=0.2, hspace=0.2)

    plt.show()

def main():
    rank_feat_for_model('classification_', 'ranking')
    rank_feat_for_model('regression_', 'ranking')


# init
if __name__ == '__main__':
    main()