# https://github.com/alisoltanirad/Dialogue-Act-Type
# Dependencies: nltk
import ssl
import nltk

def main():
    classify_act_type(get_posts())


def classify_act_type(posts):
    data_set = [(extract_features(post.text), post.get('class'))
                for post in posts]


def extract_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


def get_posts():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()
    return posts


if __name__ == '__main__':
    main()