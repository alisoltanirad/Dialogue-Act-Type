# https://github.com/alisoltanirad/Dialogue-Act-Type
# Dependencies: nltk
import ssl
import nltk

def main():
    classify_act_type(get_posts())


def classify_act_type(posts):
    pass


def get_posts():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()
    return posts


if __name__ == '__main__':
    main()