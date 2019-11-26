from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

class Dictionaries:
    """
    This class have different dictionaries used in the project
    """

    #Emoji dictionaries
    """
    Reference: Wood, I., and Sebastian Ruder. "Emoji as emotion tags for tweets." In Proceedings of the Emotion and Sentiment Analysis Workshop LREC2016, Portorož, Slovenia, pp. 76-79. 2016.   
    """
    emoji_dict = {
        "anger": ['😬', '😠', '😐', '😑', '😠', '😡', '😖', '😤', '😾'],
        "fear": ['😅', '😦', '😧', '😱', '😨', '😰', '🙀'],
        "joy": ['😀', '😂', '😃', '😄', '😆', '😇', '😉', '😊', '😋', '😌', '😍', '😎', '😏', '🌞', '😘', '😜', '😝', '😛', '😺', '😸', '😹', '😻', '😼', '💖', '💕', '😁', '♥', '❤', '☺'],
        "sadness": ['😔', '😕', '😫', '😩', '😢', '😥', '😪', '😓', '😭', '😿', '💔', '☹']
    }

    #Emoticons dictionary
    """
    Annotation based on # nrc_emotion_lexicon_wordlevel, ekphrasis emoticons.py class and self annotation
    Reference:
    1) Mohammad, Saif M., and Peter D. Turney. "Crowdsourcing a word–emotion association lexicon." Computational Intelligence 29, no. 3 (2013): 436-465.
    2) Mohammad, Saif M., and Peter D. Turney. "Emotions evoked by common words and phrases: Using mechanical turk to create an emotion lexicon."
    In Proceedings of the NAACL HLT 2010 workshop on computational approaches to analysis and generation of emotion in text, pp. 26-34. Association for Computational Linguistics, 2010.
    3) Baziotis, Christos, Nikos Pelekis, and Christos Doulkeridis. "Datastories at semeval-2017 task 4: Deep lstm with attention for message-level and topic-based sentiment analysis." 
    In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 747-754. 2017.
    """
    #
    # Reference: ekphrasis text for emoticons, https://github.com/cbaziotis/ekphrasis/blob/master/ekphrasis/dicts/emoticons.py [highfive, laugh, heart, happy]
    # Reference: NRC Word-Emotion Association Lexicon - http://sentiment.nrc.ca/lexicons-for-research/ [annoyed, devil, surprise, kiss, angel,sad]
    # Self annotation: [tong, seallips, wink]
    emoticon_dict = {
        "anger": ['annoyed', 'devil'],
        "fear": ['surprise', 'devil'],
        "joy": ['kiss', 'angel', 'wink', 'surprise', 'tong', 'highfive', 'laugh', 'heart', 'happy'],
        "sadness": ['sad', 'annoyed', 'devil'],
        "other": ['seallips']
    }

    #Classfier dictionary
    classifier_dict = {
        'NB': GaussianNB(),
        'Log Reg': LogisticRegression(),
        'SVC': SVC(), # Default kernel = 'rbf'
        'SVC_L': SVC(kernel='linear'),
        'RFC': RandomForestClassifier()
    }

    #Regressors dictionary
    regressor_dict = {
        'LinReg': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'SVR': SVR(), # Default kernel = 'rbf'
        'SVR_L': SVR(kernel='linear'),
        'RFR': RandomForestRegressor()

    }

    # Classifier dictionary
    feat_anal_classifier_dict = {
        'Log Reg': LogisticRegression(), #'SVC': SVC(), # no class_f
        'SVC_L': SVC(kernel='linear'),
        'RFC': RandomForestClassifier()
    }

    # Regressors dictionary
    feat_anal_regressor_dict = {
        'LinReg': LinearRegression(),
        'Lasso': Lasso(alpha=0.1), #'SVL': SVR(),  # Default kernel = 'rbf' # no class_f
        'SVR_L': SVR(kernel='linear'),
        'RFR': RandomForestRegressor()
    }

    #Vectorizer dictionary with unigram and bigram
    vectorizer_dict = {
        'Count_Vect_Unigram': CountVectorizer(max_features=None, min_df= 5, max_df=0.8),
        'TfIdf_Vect_Unigram': TfidfVectorizer(max_features=None, min_df= 5, max_df=0.8),
        'Count_Vect_Bigram': CountVectorizer(max_features=None, ngram_range=(1, 2), min_df= 5, max_df=0.8),
        'TfIdf_Vect_Bigram': TfidfVectorizer(max_features=None, ngram_range=(1, 2), min_df= 5, max_df=0.8)
    }

    # Resampler dictionary: SMOTE - oversampling ,TomekLinks - undersampling, SMOTETomek - combination of over- and undersampling
    resampler_dict = {
        'SMOTE': SMOTE(ratio='minority'),
        'TomekLinks': TomekLinks(ratio='majority'),
        'SMOTETomek': SMOTETomek(ratio='auto')
    }

    #scoring dictionary
    scoring = {
        'c': 'f1_macro',
        'r': 'neg_mean_squared_error'
    }


    # Dicts and lists for accessing names
    emo_dict = {1: 'anger', 2: 'fear', 3: 'joy', 4: 'sadness'}
    anal_dict = {'m': 'model', 'v' : 'vect', 'f': 'feat'}

    class_k_dict = {
        0: 'all_in', # 0 is an arbitrary value to indicate 'all features'
        1500: '1500',
        1000: '1000',
        500: '500',
        250: '250',
        100: '100'}

    reg_k_dict = {
        0: 'all_in', # 0 is an arbitrary value to indicate 'all features'
        650: '650',
        400: '400',
        200: '200',
        100: '100',
        50: '50'
    }

    classifiers = ['NB', 'LogReg', 'SVM_Rbf', 'SVM_L', 'RFC']
    regressors = ['LinReg','Lasso', 'SVM_Rbf', 'SVM_L', 'RFR']
    regressors_without_linreg = ['Lasso', 'SVM_Rbf', 'SVM_L', 'RFR']
    datasets = ['original', 'SMOTE', 'TomekLinks', 'SMOTETomek']
    #datasets = ['original', 'SMOTE', 'Tomek', 'SMOTETomek']
    vectorizers = ['CV_uni', 'Tf_idf_uni', 'CV_uniBi', 'Tf_idf_uniBi']


    trans_feature_list= ['word_count','tweet_length','emo_word_count','emo_query_word_count','hashtag_count',
                         'emo_hashtag_count','emoji_emoticon_count','emo_emoji_emoticon_count',
                         'exp_punct_count','punct_period_count','user_mention_count',
                         'all_caps_presence','emo_avg_intensity','emo_intensity_per_word_count']

    dataset_dict ={
        1: 'original',
        2: 'SMOTE',
        3: 'TomekLinks',
        4: 'SMOTETomek'
    }
    class_estimator_dict = {
        1: 'NB',
        2: 'LogReg',
        3: 'SVC',
        4: 'SVC_L',
        5:  'RFC'
    }
    reg_estimator_dict = {
        1: 'LinReg',
        2: 'Lasso',
        3: 'SVR',
        4: 'SVR_L',
        5: 'RFR'
    }
    vect_name_dict ={
        1: 'CV_Uni',
        2: 'TfIdf_Uni',
        3: 'CV_Uni_Bi',
        4: 'TfIdf_Uni_Bi'
    }
