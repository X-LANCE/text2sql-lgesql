#coding=utf8
PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'

POS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
    'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

COLUMN_TYPES = ['others', 'text', 'time', 'number', 'boolean']
TYPE_CONTROL = ['NONE', 'AGG', 'MORE', 'MOST', 'NUMBER', 'YEAR']
NER_4 = ['NONE', 'PER', 'LOC', 'ORG', 'MISC']
NER_18 = ['NONE', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
    'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
    'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
MAX_RELATIVE_DIST = 3
RELATIVE_POSITION = ['no-relation', 'question-question'] + ['question-dist-' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
    ['column-column', 'column-identity', 'same-table', 'foreign-key-f', 'foreign-key-r'] + \
    ['table-table', 'table-identity', 'foreign-key-tab-f', 'foreign-key-tab-r', 'foreign-key-tab-b'] + \
    ['column-table', 'table-column', 'primary-key-f', 'primary-key-r', 'belongs-to-f', 'belongs-to-r'] + \
    ['question-column-exactmatch', 'question-column-partialmatch', 'question-column-nomatch', 'value-column',
    'column-question-exactmatch', 'column-question-partialmatch', 'column-question-nomatch', 'column-value'] + \
    ['question-table-exactmatch', 'question-table-partialmatch', 'question-table-nomatch',
    'table-question-exactmatch', 'table-question-partialmatch', 'table-question-nomatch']

class Vocab():

    def __init__(self, padding=True, unk=True, bos=False, min_freq=1, file_path=None, items=None, default=UNK, specials=[]):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        self.default = default
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK
        if bos:
            idx = len(self.word2id)
            self.word2id[BOS], self.id2word[idx] = idx, BOS
            self.word2id[EOS], self.id2word[idx + 1] = idx + 1, EOS
        for w in specials:
            if w not in self.word2id:
                idx = len(self.word2id)
                self.word2id[w], self.id2word[idx] = idx, w
        if file_path:
            self.from_file(file_path, min_freq=min_freq)
        elif items:
            self.from_items(items)
        assert (self.default is None) or (self.default in self.word2id)

    def from_file(self, file_path, min_freq=1):
        with open(file_path, 'r', encoding='utf-8') as inf:
            for line in inf:
                line = line.strip()
                if line == '': continue
                line = line.split('\t') # ignore count or frequency
                if len(line) == 1:
                    word, freq = line[0], min_freq
                else:
                    assert len(line) == 2
                    word, freq = line
                word = word.lower()
                if word not in self.word2id and int(freq) >= min_freq:
                    idx = len(self.word2id)
                    self.word2id[word] = idx
                    self.id2word[idx] = word

    def from_items(self, items):
        for item in items:
            if item not in self.word2id:
                idx = len(self.word2id)
                self.word2id[item] = idx
                self.id2word[idx] = item

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, key):
        """ If self.default is None, it means we do not allow out of vocabulary token;
        If self.default is not None, we get the idx of self.default if key does not exist.
        """
        if self.default is None:
            return self.word2id[key]
        else:
            return self.word2id.get(key, self.word2id[self.default])
