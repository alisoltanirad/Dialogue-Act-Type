# https://github.com/alisoltanirad/Dialogue-Act-Type
# Dependencies: nltk
import ssl
import nltk

def main():
    classify_act_type(get_posts())


def classify_act_type(posts):
    train_set, test_set = preprocess_data(posts)


def preprocess_data(posts):
    data_set = [(extract_features(post.text), post.get('class'))
                for post in posts]
    return split_corpus(data_set)


def split_corpus(data_set):
    test_size = len(data_set) / 4
    train_set, test_set = data_set[test_size:], data_set[:test_size]
    return train_set, test_set


def extract_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


def get_posts():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('nps_chat')
    return nltk.corpus.nps_chat.xml_posts()


if __name__ == '__main__':
    main()