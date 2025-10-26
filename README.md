# Word2Vec

A word embedding library based on the continuous bag of words model from word2vec.

The ```Word2Vec``` class generates word embeddings from the provided corpus using unsupervised learning.

```cpp
/* Create Word2Vec Instance */

std::vector<std::string> corpus = { "a", "very", "long", "list", "of", "words", "for", "training" };
std::size_t contextWindowSize = 2;
std::size_t negativeSampleCount = 4;
std::size_t embedDimensions = 20;

Word2Vec myWord2Vec(
    corpus,
    contextWindowSize,
    negativeSampleCount,
    embedDimensions
);
```

```cpp
/* Train One Epoch */

myWord2Vec.trainStochasticEpoch(learningRate);
```

```cpp
/* Post Process Embeddings */

myWord2Vec.postProcess();
```

```cpp
/* Find Similar Words */

std::string word = "cat";
int n = 3;

std::vector<std::string> nMostSimilarToWord = myWord2Vec.findSimilarToWord(word, n);
```

```cpp
/* View Embedding Vectors */

std::vector<float> kingEmbedding = myWord2Vec.getEmbedding("king");
```

```cpp
/* Compose Embedding Vectors */

std::vector<float> compositionEmbedding;

for (int i = 0;i<kingEmbedding.size();i++) {
    compositionEmbedding.push_back(kingEmbedding[i] + womanEmbedding[i] - manEmbedding[i]);
}

int n = 5;

std::vector<std::string> nMostSimilarToComposition = myWord2Vec.findSimilarToEmbedding(compositionEmbedding, n);
```

```cpp
/* Save Model Parameters */

myWord2Vec.save("path/to/backup");
```

```cpp
/* Load Model Parameters */

myWord2Vec.load("path/to/backup");
```