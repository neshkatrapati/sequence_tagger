# Character and Word-level language modeling RNN

Code taken from Pytorch examples and then modified.
This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.

Sample Telugu Data Provided in data/tel
wikitext english is also provided.

```bash
python main.py --cuda --tied --epochs 30   # LM on 2,00,000 telugu sentences - 112 test perplexity
python generate.py                      # Generate samples from the trained LSTM model.
```
