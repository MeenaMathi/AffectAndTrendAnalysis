from __future__ import division

from feature_engineering.nrc_file_data_extractor import Nrc_File_Data_Extractor
from model_selection.dictionaries import Dictionaries
import pandas as pd
from pathlib import Path
import re
import os

#Ref: https://saifmohammad.com/WebPages/AccessResource.htm

class Feature_Transformer:
    """
    This class transforms the features into machine understandable features.
    """

    def find_emo_query_words(tokens, emotion):
        """
        Method to find and count the query words in text used in Sem Eval 2018 task 1 depending upon the emotion
        Reference: Mohammad, Saif, Felipe Bravo-Marquez, Mohammad Salameh, and Svetlana Kiritchenko.
        "Semeval-2018 task 1: Affect in tweets." In Proceedings of The 12th International Workshop on Semantic Evaluation, pp. 1-17. 2018.
        """
        parent_dir = Path.cwd().parent
        text_words = tokens
        emotion_query_word_count = 0
        for word in text_words:
            word = word.strip()
            if emotion == 'anger':
                anger_words = open(parent_dir.joinpath('documents','anger.words'), encoding='utf-8').read()
                if (re.search(r'\b' + re.escape(word.lower()) + r'\b', anger_words, re.MULTILINE)):
                    emotion_query_word_count = emotion_query_word_count + 1
            elif emotion == 'fear':
                fear_words = open(parent_dir.joinpath('documents','fear.words'), encoding='utf-8').read()
                if (re.search(r'\b' + re.escape(word.lower()) + r'\b', fear_words, re.MULTILINE)):
                    emotion_query_word_count = emotion_query_word_count + 1
            elif emotion == 'joy':
                joy_words = open(parent_dir.joinpath('documents','joy.words'), encoding='utf-8').read()
                if (re.search(r'\b' + re.escape(word.lower()) + r'\b', joy_words, re.MULTILINE)):
                    emotion_query_word_count = emotion_query_word_count + 1
            elif emotion == 'sadness':
                sadness_words = open(parent_dir.joinpath('documents','sadness.words'), encoding='utf-8').read()
                if (re.search(r'\b' + re.escape(word.lower()) + r'\b', sadness_words, re.MULTILINE)):
                    emotion_query_word_count = emotion_query_word_count + 1
        return emotion_query_word_count

    #
    def find_emo_words(tokens, emotion):
        """
        A method to find and count the words that are present in NRC emotion words depending upon the emotion
        """
        nrc_emo_words = Nrc_File_Data_Extractor.get_nrc_word_level_emotion(emotion)
        emo_word_count = 0
        for word in tokens:
            if word in nrc_emo_words:
                emo_word_count = emo_word_count + 1
        return emo_word_count

    def find_emo_hashtags(items, emotion):
        """
        A method to find and count the hashtags that are present in NRC hashtag file depending upon the emotion
        """
        emo_hashtag_count = 0
        if items.__len__() == 0:
            emo_hashtag_count = 0
        else:
            nrc_emo_hashtags_df = Nrc_File_Data_Extractor.get_nrc_hashtag_emotion(emotion)
            nrc_emo_hashtags = list(nrc_emo_hashtags_df.crae) #'crae' is the column index of emotion words
            #Remove the hashtags from the words in nrc hastags
            nrc_words = []
            for nrc_hashtag in nrc_emo_hashtags:
                #print(nrc_hashtag, nrc_hashtag.replace('#', ''))
                nrc_hashtag = str(nrc_hashtag)
                nrc_words.append(nrc_hashtag.replace('#', ''))

            for word in items:
                # Remove the space from the hashtag words to compare with nrc hashtag words
                word = word.replace(' ','')
                if word in nrc_words:
                    emo_hashtag_count = emo_hashtag_count + 1
        return emo_hashtag_count

    def find_emo_emojis(items, emotion):
        """
        A method to find and count the emojis present in the emoji list defined in Wood, I., and Sebastian Ruder.
        "Emoji as emotion tags for tweets." In Proceedings of the Emotion and Sentiment Analysis Workshop LREC2016, Portorož, Slovenia, pp. 76-79. 2016.
        """
        emo_emoji_count=0
        if items.__len__() == 0:
            emo_emoji_count = 0
        else:
            emo_emojis = Dictionaries.emoji_dict.get(emotion)
            for word in items:
                if word in emo_emojis:
                    emo_emoji_count = emo_emoji_count + 1
        return emo_emoji_count

    def find_emo_emoticons(items, emotion):
        """
        A method to find and count the emoticons that are annotated partly based on Ekphrasis and NRC Emotion words and partly based on self annotation.
        """
        emo_emoticon_count = 0
        if items.__len__() == 0:
            emo_emoticon_count = 0
        else:
            emo_emoticons = Dictionaries.emoticon_dict.get(emotion)
            for word in items:
                if word in emo_emoticons:
                    emo_emoticon_count = emo_emoticon_count + 1
        return emo_emoticon_count

    def find_emo_intensity(tokens, emotion, total_word_count):
        """
        A method to find the average emotion intensity per tweet and per word count based on data found in NRC affect intensity file, depending upon the emotion
        """
        dataframe = Nrc_File_Data_Extractor.get_nrc_affect_intensity(emotion)
        nrc_emo_words = list(dataframe.term)
        emo_word_count = 0
        emo_intensity_sum = 0.0
        emo_avg_intensity = 0.0
        emo_intensity_per_word_count = 0.0
        for word in tokens:
            if word in nrc_emo_words:
                index = dataframe[dataframe.term == word].index[0]
                emo_intensity_sum = emo_intensity_sum + dataframe.loc[index, 'score']
                emo_word_count = emo_word_count + 1
        if emo_word_count != 0.0:
            emo_avg_intensity = emo_intensity_sum / emo_word_count
        if total_word_count != 0.0:
            emo_intensity_per_word_count = emo_intensity_sum / total_word_count
        return emo_avg_intensity, emo_intensity_per_word_count

    def perform(dataframe, emotion, train_or_test, task_name):
        """
        A method to do all the feature transformation for the given emotion depending upon the task and data.
        """
        parent_dir = Path.cwd().parent
        pickle_folder_name = parent_dir.joinpath('new_results', 'pickle_files_feat_eng')
        # Create a directory, if it doesn't exist
        if not os.path.exists(pickle_folder_name):
            os.makedirs(pickle_folder_name)

        pickle_file_name = pickle_folder_name.joinpath(emotion + '_' + task_name + '_' + train_or_test + '_feat_transform_df.pkl')
        # Check the existence of the file and if the file is not empty
        if os.path.exists(pickle_file_name) and os.path.getsize(pickle_file_name) > 0:
            features_dataframe = pd.read_pickle(pickle_file_name)
            return features_dataframe
        else:
            # Create a new data frame
            col_names = ['word_count', 'tweet_length', 'emo_word_count', 'emo_query_word_count',
                         'hashtag_count', 'emo_hashtag_count', 'emoji_emoticon_count','emo_emoji_emoticon_count',
                         'exp_punct_count', 'punct_period_count', 'user_mention_count', 'all_caps_presence',
                         'emo_avg_intensity', 'emo_intensity_per_word_count']
            features_dataframe = pd.DataFrame(columns=col_names)

            for row in dataframe.itertuples(index = False):
                tokens = row[0]
                hashtags = row[3]
                emojis  = row[5]
                emoticons = row[6]

                #Find the number of Query_words, word_emotion, hashtag_emotion, emoji_emotion, emoticon_emotion
                emo_query_word_count = Feature_Transformer.find_emo_query_words(tokens, emotion)
                emo_word_count = Feature_Transformer.find_emo_words(tokens, emotion)
                emo_hashtag_count = Feature_Transformer.find_emo_hashtags(hashtags, emotion)
                emo_emoji_count = Feature_Transformer.find_emo_emojis(emojis, emotion)
                emo_emoticon_count = Feature_Transformer.find_emo_emoticons(emoticons, emotion)
                emo_avg_intensity, emo_intensity_per_word_count = Feature_Transformer.find_emo_intensity(tokens, emotion, row[1])


                new_data = {'word_count': row[1],
                            'tweet_length': row[2],
                            'emo_word_count': emo_word_count,
                            'emo_query_word_count': emo_query_word_count,
                            'hashtag_count': row[4],
                            'emo_hashtag_count': emo_hashtag_count,
                            'emoji_emoticon_count': row[7],
                            'emo_emoji_emoticon_count': emo_emoji_count + emo_emoticon_count,
                            'exp_punct_count': row[8],
                            'punct_period_count': row[9],
                            'user_mention_count': row[10],
                            'all_caps_presence': row[11],
                            'emo_avg_intensity': emo_avg_intensity,
                            'emo_intensity_per_word_count': emo_intensity_per_word_count
                            }
                features_dataframe.loc[len(features_dataframe)] = new_data

            #Simply impute the dataframe null values with 0s
            features_dataframe[features_dataframe.isnull()] = 0.0

            features_dataframe.to_pickle(pickle_file_name)
            print(features_dataframe)
            return features_dataframe
