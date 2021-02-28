import numpy as np
from collections import defaultdict
from itertools import combinations


class BoolSearch:
    def __init__(self, documents, tokenizer=None):
        self.qa_dict = documents
        self.documents = [item['question'] for item in documents]
        self.tokenizer = tokenizer
        self.dic_word_doc = defaultdict(list)   # word2docid 每个词对应那个doc
        self.doc_map = {}   # docid2doc
        self.dic_word_id = defaultdict(int)
        self.matrix = self._build_matrix()

    def _build_rev_dic(self):
        for doc_id, item_ in enumerate(self.qa_dict):
            self.doc_map[doc_id] = item_

            words = self.tokenizer(item_["question"])
            for word in words:
                self.dic_word_doc[word].append(doc_id)

    def _build_matrix(self):
        self._build_rev_dic()

        word_num = len(self.dic_word_doc)
        doc_num = len(self.documents)
        matrix = np.zeros((word_num, doc_num)).astype(np.int16)

        for word_id, (word, doc_ids) in enumerate(self.dic_word_doc.items()):
            for doc_id in self.doc_map:
                if doc_id in doc_ids:
                    matrix[word_id, doc_id] = 1

            self.dic_word_id[word] = word_id

        return matrix

    def _get_vector_inter(self, word_ids):
        vectors = self.matrix[word_ids]
        vector_inter = np.where(vectors.sum(axis=0) == len(word_ids), 1, 0)
        return vector_inter

    def _get_vector(self, word_ids, top_n=10):
        if top_n == 0:
            return []

        top_n = len(word_ids[:top_n])

        comb_ids = list(combinations(range(len(word_ids)), top_n))
        for ids in comb_ids:
            word_ids_f = [word_ids[idx] for idx in ids]
            vector_inter = self._get_vector_inter(word_ids_f)

            if max(vector_inter) == 1:
                return vector_inter

        return self._get_vector(word_ids, top_n-1)

    def get_top_n(self, query, n=10):
        words = self.tokenizer(str(query))
        word_ids = [self.dic_word_id[word] for word in words if word in self.dic_word_id]

        if not word_ids:
            return []

        vector = self._get_vector(word_ids)

        if len(vector) == 0:
            return []

        docs = [self.doc_map[i] for i, idx in enumerate(vector) if idx == 1]
        return docs[:n]
