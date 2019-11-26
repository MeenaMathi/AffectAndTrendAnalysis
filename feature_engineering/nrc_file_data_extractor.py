import pandas as pd
from pathlib import Path


class Nrc_File_Data_Extractor:
    """
    A class to access NRC files and retrieve data depending upon the emotion

    To download NRC Word-Emotion Association Lexicons:
    http://saifmohammad.com/WebPages/lexicons.html
    http://sentiment.nrc.ca/lexicons-for-research/
    """
    parent_dir = Path.cwd().parent

    nrc_affect_intensity = pd.read_csv(parent_dir.joinpath('documents','NRC-AffectIntensity-Lexicon.txt'),delimiter="\t", encoding="utf-8-sig")
    nrc_word_level_emotion = pd.read_csv(parent_dir.joinpath('documents','NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'), delimiter="\t", encoding="utf-8-sig")
    nrc_hashtag_emotion = pd.read_csv(parent_dir.joinpath('documents','NRC-Hashtag-Emotion-Lexicon-v0.2.txt'), delimiter="\t", encoding="utf-8-sig")

    def get_nrc_affect_intensity(emotion):
        """
        A method to retrieve affect intensity from the NRC file depending upon the affect
        Reference:Mohammad, Saif M. "Word affect intensities." arXiv preprint arXiv:1704.08798 (2017).
        """
        if emotion == 'anger':
            dataframe = Nrc_File_Data_Extractor.nrc_affect_intensity
            dataframe = dataframe[dataframe['AffectDimension'] == 'anger']
            return dataframe
        if emotion == 'fear':
            dataframe = Nrc_File_Data_Extractor.nrc_affect_intensity
            dataframe = dataframe[dataframe['AffectDimension'] == 'fear']
            return dataframe
        if emotion == 'joy':
            dataframe = Nrc_File_Data_Extractor.nrc_affect_intensity
            dataframe = dataframe[dataframe['AffectDimension'] == 'joy']
            return dataframe
        if emotion == 'sadness':
            dataframe = Nrc_File_Data_Extractor.nrc_affect_intensity
            dataframe = dataframe[dataframe['AffectDimension'] == 'sadness']
            return dataframe

    def get_nrc_word_level_emotion(emotion):
        """
        A method to retrieve word level emotions from the NRC file depending upon the affect
        Reference: 1) Mohammad, Saif M., and Peter D. Turney. "Crowdsourcing a word–emotion association lexicon." Computational Intelligence 29, no. 3 (2013): 436-465.
        2) Mohammad, Saif M., and Peter D. Turney. "Emotions evoked by common words and phrases: Using mechanical turk to create an emotion lexicon."
        In Proceedings of the NAACL HLT 2010 workshop on computational approaches to analysis and generation of emotion in text, pp. 26-34. Association for Computational Linguistics, 2010.
        """
        if emotion == 'anger':
            dataframe = Nrc_File_Data_Extractor.nrc_word_level_emotion
            dataframe = dataframe[dataframe['0'] == 1]
            dataframe = dataframe[dataframe['anger']== 'anger']
            return list(dataframe.aback) # 'aback' is the column index of words column
        if emotion == 'fear':
            dataframe = Nrc_File_Data_Extractor.nrc_word_level_emotion
            dataframe = dataframe[dataframe['0']== 1]
            dataframe = dataframe[dataframe['anger']== 'fear']
            return list(dataframe.aback)  # 'aback' is the column index of words column
        if emotion == 'joy':
            dataframe = Nrc_File_Data_Extractor.nrc_word_level_emotion
            dataframe = dataframe[dataframe['0']== 1]
            dataframe = dataframe[dataframe['anger']== 'joy']
            return list(dataframe.aback)  # 'aback' is the column index of words column
        if emotion == 'sadness':
            dataframe = Nrc_File_Data_Extractor.nrc_word_level_emotion
            dataframe = dataframe[dataframe['0']== 1]
            dataframe = dataframe[dataframe['anger']== 'sadness']
            return list(dataframe.aback)  # 'aback' is the column index of words column

    def get_nrc_hashtag_emotion(emotion):
        """
        A method to retrieve hashtag emotions from the NRC file depending upon the affect
        Reference: 1) Mohammad, Saif M. "# Emotional tweets." In Proceedings of the First Joint Conference on Lexical and Computational Semantics-
        Volume 1: Proceedings of the main conference and the shared task,
        and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation, pp. 246-255. Association for Computational Linguistics, 2012.
        2) Mohammad, Saif M., and Svetlana Kiritchenko. "Using hashtags to capture fine emotion categories from tweets." Computational Intelligence 31, no. 2 (2015): 301-326.
        """
        if emotion == 'anger':
            dataframe = Nrc_File_Data_Extractor.nrc_hashtag_emotion
            dataframe = dataframe[dataframe['anticipation'] == 'anger']
            return dataframe
        if emotion == 'fear':
            dataframe = Nrc_File_Data_Extractor.nrc_hashtag_emotion
            dataframe = dataframe[dataframe['anticipation'] == 'fear']
            return dataframe
        if emotion == 'joy':
            dataframe = Nrc_File_Data_Extractor.nrc_hashtag_emotion
            dataframe = dataframe[dataframe['anticipation'] == 'joy']
            return dataframe
        if emotion == 'sadness':
            dataframe = Nrc_File_Data_Extractor.nrc_hashtag_emotion
            dataframe = dataframe[dataframe['anticipation'] == 'sadness']
            return dataframe
        if emotion == 'all_four':
            dataframe = Nrc_File_Data_Extractor.nrc_hashtag_emotion
            df_anger = (dataframe[dataframe['anticipation'] == 'anger'])
            df_fear = dataframe[dataframe['anticipation'] == 'fear']
            df_joy = dataframe[dataframe['anticipation'] == 'joy']
            df_sadness = dataframe[dataframe['anticipation'] == 'sadness']
            df = [df_anger,df_fear,df_joy,df_sadness]
            dataframe = pd.concat(df)
            return dataframe


"""
#### TESTING
anger_words = open(r'./documents/anger.words',encoding='utf-8').read()
fear_words = open(r'./documents/fear.words',encoding='utf-8').read()
joy_words = open(r'./documents/joy.words',encoding='utf-8').read()
sadness_words = open(r'./documents/sadness.words',encoding='utf-8').read()

dataframe = Nrc_File_Data_Extractor.get_nrc_hashtag_emotion('anger')
print('HASHTAGS')
print(dataframe.shape)

dataframe = Nrc_File_Data_Extractor.get_nrc_affect_intensity('anger')
print('INTENSITY')
print(dataframe.shape)

list = Nrc_File_Data_Extractor.get_nrc_word_level_emotion('anger')
print('EMO-WORDS')
print(list)
"""
