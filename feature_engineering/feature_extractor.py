from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import emoji

class FeatureExtractor:
    """
    This class extracts the important features from the text using ekphrasis and emoji.
    Reference:
    1) Baziotis, Christos, Nikos Pelekis, and Christos Doulkeridis. "Datastories at semeval-2017 task 4: Deep lstm with attention for message-level and topic-based sentiment analysis."
    In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 747-754. 2017.
    2) Emoji module: https://github.com/kyokomi/emoji
    """
    # Ekphrasis: Define TextPreProcessor attributes
    text_processor = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=True,  # fix html and other tokens
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        unpack_contractions=True,  # (can't -> can not)
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=False).tokenize,  # preserve the case
        # tokenizer=ws_tokenizer, #Tokenizer dealt separately
        dicts=[emoticons]
    )

    def remove_annotation_tags(text):
        # Remove all Ekphrasis tags used for normalization and annotation
        annotation_tags = ['<url>', '<email>', '<percent>', '<money>', '<phone>', '<time>', '<date>', '<number>',
                           '<elongated>', '<repeated>', '<emphasis>', '<censored>', '<hashtag>',
                           '</hashtag>', '<user>', '<allcaps>', '</allcaps>']
        text_words = text.split()
        unremoved_words = [word for word in text_words if word not in annotation_tags]
        result_text = ' '.join(unremoved_words)
        return result_text


    def check_presence(list):
        # Check if the list is empty
        if list.__len__() > 0:
            return 1
        else:
            return 0


    def find_emojis_and_convert_to_text(text):
        # Collect the emoji seperately and add as text in the original text.
        text_words = text.split()
        new_word_list = []
        emoji_words = []
        for word in text_words:
            word = word.strip()
            for char in word:
                if char in emoji.UNICODE_EMOJI:
                    emoji_words.append(word)
                    word = emoji.demojize(word)
                    new_word_list.append(emoji.demojize(word))
                    break
                else:
                    new_word_list.append(word)
                    break
        result_text = ' '.join(new_word_list)
        return emoji_words, result_text
