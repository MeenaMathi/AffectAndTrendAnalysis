from feature_engineering.feature_extractor import FeatureExtractor
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import re
import os


class Preprocessor:
    """
    A class for doing various preprocessing tasks depending upon the task, emotion and data given. Parallely feature extraction is also performed
    """
    def perform(dataset, emotion, train_or_test, task_name):
        parent_dir = Path.cwd().parent
        pickle_dir = parent_dir.joinpath('new_results', 'pickle_files_feat_eng')

        # Create a directory for storing pipelines, if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        pickle_file_name = pickle_dir.joinpath(emotion + '_' + task_name + '_' + train_or_test + '_preprocess_df.pkl')

        # Check the existence of the file and if the file is not empty
        if os.path.exists(pickle_file_name) and os.path.getsize(pickle_file_name) > 0:
            preprocessor_dataframe = pd.read_pickle(pickle_file_name)
            return preprocessor_dataframe
        else:
            # Create a new data frame
            col_names = ['preprocessed_tokens', 'word_count', 'tweet_length', 'hashtags',
                             'hashtag_count', 'emoji', 'emoticon', 'emoji_emoticon_count',
                             'exp_punct_count', 'punct_period_count', 'user_mention_count', 'all_caps_presence',
                             'Affect Dimension','preprocessed_text']
            preprocessor_dataframe = pd.DataFrame(columns=col_names)


            for row in dataset.itertuples(index = False):
                tweet = row[1]

                ekphrasis_text =(" ".join(FeatureExtractor.text_processor.pre_process_doc(tweet)))

                #Remove repeated periods, repeated commas, RT,rt,CC,cc,line breaks
                text = re.sub('[\,]+', ',',ekphrasis_text)
                text = re.sub('[\.]+', '.', text)
                text = re.sub('\bRT\b|\bcc\b|\brt\b|\bCC\b', '', text) #only useful for test data
                text = re.sub('[\b\\n\b]', '.',text)


                # Handle hashtags, all_caps words and user handles separate
                hashtag_list = re.findall("<hashtag>(.*?)</hashtag>", text)
                all_caps_wordlist = re.findall("<allcaps>(.*?)</allcaps>", text)
                all_caps_presence = FeatureExtractor.check_presence(all_caps_wordlist)
                user_handles = re.findall("<user>", text)
                text = re.sub("<user>", '_user_', text)

                #Remove annotation tags from the text
                text = FeatureExtractor.remove_annotation_tags(text)

                #Collect emojis and emoticons
                emoticons = re.findall("<\w+>", text) #emoticons are already converted into <text> form
                emojis, text = FeatureExtractor.find_emojis_and_convert_to_text(text)


                # Handle question marks, exclamatory marks and period
                q_mark = re.findall("\?", text)
                text = re.sub("\?",'q_mark', text)
                e_mark = re.findall("!", text)
                text = re.sub("!", 'e_mark', text)
                punct_period = re.findall("\.", text)
                text = re.sub("\.", 'punct_period', text)


                # Change to lower case, do white space collapsing, remove digits from words
                text = text.lower()
                text = re.sub(r'[^\w\s]',' ', text)
                text = re.sub(r'[\s]+',' ', text)
                text = re.sub(r'[\d]+',' ', text)

                # Tokenize, do stemming, remove punctuations, remove stopwords
                tokenizer = TweetTokenizer()
                tokens = tokenizer.tokenize(text)
                stemmer = SnowballStemmer("english", ignore_stopwords=True)

                stop_words = set(stopwords.words('english'))
                clean_tokens = []
                regex = re.compile('^[a-zA-Z]+$')

                for token in tokens:
                    if (bool(re.match(regex, token))) :
                        token = stemmer.stem(token)
                        #Remove stop words, other language words, numbers and words with length < 1 and punctuations
                        if (token not in stop_words) & (len(token) > 1):
                            clean_tokens.append(token)
                if(clean_tokens.__len__() < 1):
                    clean_tokens.append("noTokens")


                new_data = {'preprocessed_tokens': clean_tokens,
                            'word_count': tweet.split().__len__(),
                            'tweet_length': len(tweet),
                            'hashtags':hashtag_list,
                            'hashtag_count': hashtag_list.__len__(),
                            'emoji': emojis,
                            'emoticon': emoticons,
                            'emoji_emoticon_count': emojis.__len__() + emoticons.__len__(),
                            'exp_punct_count': q_mark.__len__() + e_mark.__len__(),
                            'punct_period_count': punct_period.__len__(),
                            'user_mention_count': user_handles.__len__(),
                            'all_caps_presence': all_caps_presence,
                            'Affect Dimension': row[2],
                            'preprocessed_text': ' '.join(clean_tokens)
                            }
                preprocessor_dataframe.loc[len(preprocessor_dataframe)] = new_data

            preprocessor_dataframe.to_pickle(pickle_file_name)
            print(preprocessor_dataframe)
            return preprocessor_dataframe

