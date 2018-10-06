import torch.nn as nn
import torch 
from torch.autograd import Variable
from charrnnmodel import CharRNNModel

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, corpus, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.corpus = corpus
        self.char_ntoken = len(self.corpus.charmap.idx2word) + 1
       # self.char_encoder = nn.Embedding(self.char_ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp*2, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp*2, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        #    self.char_rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.char_rnn = CharRNNModel(rnn_type, len(self.corpus.charmap.idx2word) + 1,ninp,nhid,nlayers,corpus,dropout,tie_weights)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
      #  self.char_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    

    def forward(self, input, hidden):
        #print(input.size())

        ## Get Character Embeddings by running through the Character RNN
        
        # Get the number of characters in the character dictionary (charmap)
        maxchar_index = len(self.corpus.charmap.idx2word)
        maxchar_len = 0

        # Our Input is SeqLenxBatchSize this should be transformed to SeqLenxBatchSizexCharLen for each word. This is done by splitting the word into chars
        cinput = []
        for batch in input:
            cbatch = []
            for seq_item in batch:
                word = self.corpus.dictionary.idx2word[seq_item]
                chars = list(word)
                if maxchar_len < len(word):
                    maxchar_len = len(word) # Find out the max length of anyword to be used for padding                    
                char_ids = [self.corpus.charmap.word2idx[x] for x in chars] # Convert chars to ids by reverse lookup of the charmap
                cbatch.append(char_ids)
            cinput.append(cbatch)

        # Padding SeqLenxBatchSizexCharLen -> SeqLenxBatchSizexMaxCharLen
        cinput_lens = []
        for cbatch in cinput:
            cib = []
            for seq_item in cbatch:
                to_pad = [maxchar_index] * (maxchar_len - len(seq_item))
                cib.append(len(seq_item) - 1)
                seq_item += to_pad 
            cinput_lens.append(cib) # Even though we pad, we must retain lengths. Failing this, taking the absolute last step of the RNN will ruin everything.
        cinput = torch.LongTensor(cinput).cuda()

        cinput_lens = torch.LongTensor(cinput_lens).cuda()
        cinput_size = cinput.size()

        #print("Before transform", cinput.size())

        
        # Transform SeqLenxBatchSizexMaxCharLen -> MaxCharLenx(SeqLen*BatchSize) 
        # This is done to give the CharRnn CharSeqLen*CharBatchSize where, CharSeqLen = MaxCharLen and CharBatchSize = (SeqLen*BatchSize).

        transformed_cinput = cinput.view(cinput_size[0]*cinput_size[1], -1) # SeqLenxBatchSizexMaxCharLen -> (SeqLen*BatchSize)xMaxCharLen
        cinput_lens = cinput_lens.view(cinput_size[0]*cinput_size[1]) # Transform Input Lengths into a long 1-D array
        transformed_cinput = transformed_cinput.permute(1,0) # Flip the (SeqLen*BatchSize)xMaxCharLen -> MaxCharLenx(SeqLen*BatchSize)
     
        
        chidden = self.char_rnn.init_hidden(cinput_size[0]*cinput_size[1])
        last_output, chidden = self.char_rnn(transformed_cinput, chidden, cinput_lens) # Pump it through the CharRNN and get the output (SeqLen*BatchSize)xEmSize

        last_output = last_output.view(cinput_size[0], cinput_size[1],-1) # Transform this back to SeqLenxBatchSizexEmSize

        emb = self.drop(self.encoder(input))
        emb = torch.cat([last_output, emb], 2) # Concatenate Char and Word Embeddings. to give 2*EmSize embeddings

        #print("Embedding size", emb.size())
        output, hidden = self.rnn(emb, hidden)
        #output, hidden = self.rnn(last_output, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
