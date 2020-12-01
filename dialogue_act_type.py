# https://github.com/alisoltanirad/Dialogue-Act-Type
# Dependencies: nltk
import ssl
import json
import nltk


class ActTypeClassifier:

    def __init__(self):
        self._train_set, self._test_set = self._get_data_set()
        self.classifier = nltk.NaiveBayesClassifier.train(self._train_set)

    def classify(self):
        pass

    def evaluate(self):
        pass

    def _get_data_set(self):
        pass

    def _preprocess_data(self):
        pass

    def _split_corpus(self):
        pass

    def _extract_features(self):
        pass

    def _get_informative_tokens(self):
        pass


def main():
    classify_act_type(get_posts())


def classify_act_type(posts):
    train_set, test_set = preprocess_data(posts)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    evaluate_classifier(classifier, test_set)


def evaluate_classifier(classifier, test_set):
    print('Evaluation\n\t- Accuracy: {:.2%}'.format(
        nltk.classify.accuracy(classifier, test_set)))


def preprocess_data(posts):
    data_set = [(extract_features(post.text), post.get('class'))
                for post in posts]
    return split_corpus(data_set)


def split_corpus(data_set):
    test_size = round(len(data_set) / 4)
    train_set, test_set = data_set[test_size:], data_set[:test_size]
    return train_set, test_set


def extract_features(post):
    features = {
        'starts with': post[0],
        'ends with': post[-1]
    }
    for char in "?@#'_":
        features['contains({})'.format(char)] = (char in post)
    for word in get_informative_tokens():
        features['contains({})'.format(word)] = (word in post)
    return features


def get_informative_tokens():
    with open('informative_tokens.txt') as token_list:
        return json.load(token_list)


def get_posts():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('nps_chat')
    return nltk.corpus.nps_chat.xml_posts()


if __name__ == '__main__':
    main()