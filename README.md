# Word2Vec

A word embedding library based on the continuous bag of words model from word2vec.

The ```Word2Vec``` class generates word embeddings from the provided corpus using unsupervised learning.

```
/* Create Word2Vec Instance */

std::vector<std::string> corpus = { "a", "very", "long", "list", "of", "words", "for", "training" };
std::size_t contextWindowSize = 2;
unsigned int embedDimension = 20;

Word2Vec myWord2Vec(
    corpus,
    contextWindowSize,
    embedDimension
);
```

```
/* Batch Train */

int batchSize = 5;
float learningRate = 0.1;

myWord2Vec.train(
    batchSize,
    learningRate
);
```

```
/* Find Similar Words */

std::string word = "cat";
int n = 3;

std::vector<std::string> nMostSimilarToWord = myWord2Vec.findSimilar(word, n);
```