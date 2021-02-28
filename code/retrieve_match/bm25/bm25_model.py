import math
import numpy as np


class BM25:
    """Best Match 25 github reference: https://github.com/dorianbrown/rank_bm25

    """
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)  # Number of documents in the corpus.
        self.avgdl = 0  # Average number of terms for documents in the corpus.
        self.doc_freqs = []     # dic中储存一个文档中每个词和出现了该词的文档数量
        self.idf = {}   # inverse Document Frequency per term.
        self.doc_len = []   # Number of terms per document. So [3] means the first document contains 3 terms.
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)
        nd = self._initialize(corpus)   # 包含该词的问题数目
        self._calc_idf(nd)  # 计算该词的权重idf

    def _initialize(self, corpus):  # 用于计算self.avgdl以及nd
        nd = {}     # word -> number of documents with word包含该词的文档数
        num_doc = 0     # 所有文档中所有词汇的总数目，也就是词总数
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}    # 用词典储存一个词在所有文档中出现的频率
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n(self, query, qa_df, n=5, calc_recall=False):
        if not calc_recall:
            assert self.corpus_size == len(qa_df), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [{"question": qa_df[i]['question'], "answer": qa_df[i]['answer']} for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []  # idf为负的词的idf会用epsilon来重置，当单词出现在大半文档中时，idf会为负
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum/len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

