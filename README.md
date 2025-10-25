# Word2Vec

A word embedding library based on the continuous bag of words model from word2vec.

The ```Word2Vec``` class generates word embeddings from the provided corpus using unsupervised learning.

```cpp
/* Create Word2Vec Instance */

std::vector<std::string> corpus = { "a", "very", "long", "list", "of", "words", "for", "training" };
int contextWindowSize = 2;
int negativeSampleCount = 4;
int embedDimension = 20;

Word2Vec myWord2Vec(
    corpus,
    contextWindowSize,
    negativeSampleCount,
    embedDimension
);
```

```cpp
/* Batch Train */

int batchSize = 5;
float learningRate = 0.1;

myWord2Vec.trainRandomBatch(
    batchSize,
    learningRate
);
```

```cpp
/* Train Stochastic Epoch */

myWord2Vec.trainStochasticEpoch(learningRate);
```

```cpp
/* Find Similar Words */

std::string word = "cat";
int n = 3;

std::vector<std::string> nMostSimilarToWord = myWord2Vec.findSimilar(word, n);
```

```cpp
/* Save Model Parameters */

myWord2Vec.save("path/to/backup");
```

```cpp
/* Load Model Parameters */

myWord2Vec.load("path/to/backup");
```