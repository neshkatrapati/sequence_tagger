import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __str__(self):
        return self.word2idx.__str__()


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.tagmap = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            num_lines = 0
            seq_len = 0
            for line in f:
                words = line.split() 
                
                for word in words:
                    word = word.split('/')
                    word,tag = '/'.join(word[:-1]), word[-1]
                    self.dictionary.add_word(word)
                    self.tagmap.add_word(tag)
                    tokens += 1

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            tags = torch.LongTensor(tokens)
            token = 0
            for line_num, line in enumerate(f):
                words = line.split()
                for word in words:
                    word = word.split('/')
                    word,tag = '/'.join(word[:-1]), word[-1]
                    ids[token] = self.dictionary.word2idx[word]
                    tags[token] = self.tagmap.word2idx[tag]
                    token += 1
                
        return ids, tags
