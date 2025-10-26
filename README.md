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
/* View Embedding Vectors */

std::vector<float> kingEmbedding = myWord2Vec.getEmbedding("king");
```

```cpp
/* Find Similar By Embedding */

std::vector<std::string> nMostSimilarToEmbedding = myWord2Vec.findSimilarToEmbedding(embedding, n);
```

```cpp
/* Find Similar Words */

std::string word = "cat";

std::vector<std::string> nMostSimilarToWord = myWord2Vec.findSimilarToWord(word, n);
```

```cpp
/* Find Similar Words To Composition */

std::vector<std::pair<std::string, float>> composition = {
    { "king", 1.0 },
    { "woman", 1.0 },
    { "man", -1.0 }
};

std::vector<std::string> nMostSimilarToComposition = myWord2Vec.findSimilarToLinearComposition(composition, n);
```

```cpp
/* Save Model Parameters */

myWord2Vec.save("path/to/backup");
```

```cpp
/* Load Model Parameters */

myWord2Vec.load("path/to/backup");
```