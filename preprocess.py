import re
from nltk.stem import SnowballStemmer

from sklearn.base import BaseEstimator, TransformerMixin


class TextPreProcessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lower=True, stem=True):
        self.lower = lower
        self.stem = stem
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # for i in range(0, X.shape[0]):
        #     comment = X.iloc[i].get('Comment')
        #     print("Comment is {}".format(comment))
        #      X.iloc[i].set_value('Comment', self._normalize(comment))
        #      print("New comment is {}".format(X.iloc[i].get('Comment')))
        # return X
        return [self._normalize(comment) for comment in X]

    def _expand_internet_slangs(self, comment):
        """
        Expand common internet slangs
        """
        comment.replace(" lol ", " laugh out loud ")
        return comment

    def _remove_eol_ws(self, comment):
        """
        Remove EOL whitespaces
        """
        comment = comment.replace("\\n", " ")
        comment = comment.replace("\\n", " ")
        comment = comment.replace("\\r", " ")
        comment = comment.replace("\\t", " ")
        return comment

    def _remove_unicodes(self, comment):
        comment = comment.replace("\\xa0", " ")
        comment = comment.replace("\\xc2", " ")
        comment = comment.replace('\\"', " ")
        return comment

    def _language_shorts(self, comment):
        """
        Expand common internet word contractions
        """
        comment = comment.replace(" u ", " you ")
        comment = comment.replace(" em ", " them ")
        comment = comment.replace(" da ", " the ")
        comment = comment.replace(" yo ", " you ")
        comment = comment.replace(" ur ", " you ")
        comment = comment.replace(" im ", " i am ")
        return comment

    def _expand_contractions(self, comment):
        """
        Expand grammetical contractions
        """
        comment = comment.replace("'s ", " is ")
        comment = comment.replace(" ain't ", " is not ")
        comment = comment.replace(" won't ", " would not ")
        comment = comment.replace(" can't ", " can not ")
        comment = comment.replace("n't ", " do not ")
        comment = comment.replace("'re ", " are ")
        comment = comment.replace("'m ", " am ")
        comment = comment.replace("'ll ", " will ")
        comment = comment.replace("'ve ", " have ")
        comment = comment.replace("'ll ", " will ")
        comment = comment.replace("'d ", " would ")
        return comment

    def _expand_and_correct_contractions(self, comment):
        comment = comment.replace(" dont ", " do not ")
        comment = comment.replace(" youre ", " you are ")
        comment = comment.replace(" aint ", " is not ")
        comment = comment.replace(" im ", " i am ")
        comment = comment.replace(" cant ", " can not ")
        comment = comment.replace(" thats ", " that is ")
        comment = comment.replace(" its ", " it is ")
        comment = comment.replace(" doesnt ", " does not ")
        comment = comment.replace(" hes ", " he is ")
        comment = comment.replace(" wont ", " would not ")
        comment = comment.replace(" arent ", " are not ")
        comment = comment.replace(" isnt ", " is not ")
        comment = comment.replace(" ill ", " i will ")
        comment = comment.replace(" ive ", " i have ")
        comment = comment.replace(" youll ", " you will ")
        comment = comment.replace(" youd ", " you would ")
        comment = comment.replace(" theres ", " there is ")
        comment = comment.replace(" wasnt ", " was not ")
        comment = comment.replace(" shouldnt ", " should not ")
        comment = comment.replace(" whats ", " what is ")
        return comment
    
    def _remove_html_tags(self, comment):
        """
        Remove all html tags.
        Online comments may contain links and iframes.
        """
        return re.sub(r'<[^<]+?>', '', comment)

    def _stem(self, comment):
        """
        Stemming: reducing token to root form.
        """
        if self.stem:
            comment = " ".join(self.stemmer.stem(word) for word in comment.split(" "))
        return comment

    def _normalize(self, comment):
        """
        Method for calling other normalization methods
        """
        comment = comment.lower() if self.lower else comment
        comment = self._remove_html_tags(comment)
        comment = self._remove_eol_ws(comment)
        comment = self._remove_unicodes(comment)
        comment = self._language_shorts(comment)
        comment = self._expand_contractions(comment)
        comment = self._expand_internet_slangs(comment)
        comment = self._expand_and_correct_contractions(comment)
        if self.stem:
            comment = self._stem(comment)
        return comment

