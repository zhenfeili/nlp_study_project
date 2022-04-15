from jieba import posseg

from keywords.textrank import TextRank
from keywords.tfidf import TFIDF

ALLOW_POS = ('ns', 'n', 'vn', 'v', 'i', 'a', 'ad', 'an')  # 被保留的词性
METHODS = ('TFIDF', 'TEXTRANK')


def get_default_stop_words():
    stop_words = set()
    with open('stop_words.txt', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if not word:
                continue
            stop_words.add(word)
    return frozenset(stop_words)


class Keyword(object):
    def __init__(self, idf_path, pos=posseg.cut, stop_words=get_default_stop_words(), idf_splitter=' ',
                 allow_pos=ALLOW_POS):
        self._pos = pos

        self._tfidf = TFIDF()
        self._tfidf.load_idf(idf_path, idf_splitter)

        self._textrank = TextRank()
        self._stop_words = set(stop_words)
        self._allow_pos = allow_pos

    @staticmethod
    def _check_input(**kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
            assert method in METHODS, '请确保method在{}中！'.format(METHODS)
        if 'title' in kwargs and 'content' in kwargs:
            title, content = kwargs['title'], kwargs['content']
            assert title or content, '请确保title和content至少一个不为空！'

    def extract(self, title, content, top_k=10, method='TFIDF', filter_stopwords=True, min_word_len=1,
                with_weight=True):
        method = method.upper()
        self._check_input(title=title, content=content, method=method)

        # 过滤停用词、不需要的词性、长度小于1的词、空白词(空格 tab 换行等)
        words = []
        text = title + ' ' + content
        for word, pos in self._pos(text):
            if filter_stopwords and word in self._stop_words or \
                    len(word) < min_word_len or \
                    pos not in self._allow_pos or \
                    not word.strip():
                continue
            words.append(word)

        if method == 'TEXTRANK':
            keywords = self._extract_by_textrank(words)
        elif method == 'TFIDF':
            keywords = self._extract_by_tfidf(words)
        else:
            raise ValueError

        if not with_weight:
            keywords = [keyword for keyword, weight in keywords]

        return keywords[:top_k]

    def _extract_by_tfidf(self, words):
        keywords = self._tfidf.compute_tfidf(words)
        return keywords

    def _extract_by_textrank(self, words):
        keywords = self._textrank.textrank(words)
        return keywords
