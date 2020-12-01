# https://github.com/alisoltanirad/Dialogue-Act-Type
# Dependencies: nltk
import ssl
import json
import nltk
import inspect


class ActTypeClassifier:

    def __init__(self):
        self._train_set, self._test_set = self._get_data()
        self._classifier = nltk.NaiveBayesClassifier.train(self._train_set)
        self.labels = self.get_labels()

    def classify(self, text):
        return self._classifier.classify(self._extract_features(text))

    def evaluate(self):
        evaluation_data = {
            'Accuracy':
                nltk.classify.accuracy(self._classifier, self._test_set),
            'Most_Informative_features':
                self._classifier.most_informative_features(50),
        }
        return evaluation_data

    def get_labels(self):
        labels = []
        for label in self._classifier.labels():
            labels.append(label)
        return labels

    def _get_data(self):
        posts = self._download_corpus()
        data_set = [(self._extract_features(post.text), post.get('class'))
                    for post in posts]
        return self._split_corpus(data_set)

    def _download_corpus(self):
        ssl._create_default_https_context = ssl._create_unverified_context
        nltk.download('nps_chat')
        return nltk.corpus.nps_chat.xml_posts()


    def _split_corpus(self, data_set):
        TEST_SIZE = round(len(data_set) / 4)
        train_set, test_set = data_set[TEST_SIZE:], data_set[:TEST_SIZE]
        return train_set, test_set

    def _extract_features(self, text):
        features = {
            'first-word': text.split()[0],
            'ends-with': text[-1]
        }
        for char in "?@#'_":
            features['contains({})'.format(char)] = (char in text)
        for word in self._get_informative_tokens():
            features['contains({})'.format(word)] = (word in text)
        return features

    def _get_informative_tokens(self):
        try:
            with open('informative_tokens.txt') as token_list:
                return json.load(token_list)
        except:
            return []


def main():
    classifier = ActTypeClassifier()
    print(
        'Evaluation\n\t- Accuracy: {:.2%}'.format(
            classifier.evaluate()['Accuracy']
        )
    )
    print(classifier.classify('Is this a question?'))


if __name__ == '__main__':
    main()