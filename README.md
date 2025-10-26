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

## Text8 Word Embeddings Example

### Training Data

The example model was trained on the full [Text8](https://www.kaggle.com/datasets/gupta24789/text8-word-embedding) corpus, an opensource 2006 Wikipedia snapshot.

### Model Parameters

The model uses 150 dimension embeddings, a &plusmn;4 word context window, and a negative sample count of 10. The learning rate is fixed at 0.02.

### Training Progression

| **Epoch** | **Similarity Tests** | **Composition Tests** |
|------------|----------------------|----------------------------------|
| **0** | dog → trepp, kunsthistorisches, cornercopia <br><br> police → eigenji, arterious, ballymena <br><br> red → nomeansno, aerospacelegacyfoundation, volli <br><br> tree → altsasu, mayottensis, saccharalis <br><br> house → plyoffs, chfp, gpj | water + frozen → auditore, utukki, outputwait <br><br> king + woman - man → lucy, queynte, urkizu <br><br> plant + tall + wood → pocomoke, drawling, citg <br><br> nature - inside → sandoy, shaffers, kinchiltun <br><br> paris + italy - france → amereon, shangugu, bivalve |
| **25** | dog → cat, dogs, baby <br><br> police → military, officer, officers <br><br> red → blue, yellow, black <br><br> tree → trees, flowers, garden <br><br> house → palace, court, home | water + frozen → dry, wet, snow <br><br> king + woman - man → queen, prince, wife <br><br> plant + tall + wood → fish, sand, water <br><br> nature - inside → temperament, mastery, grotesqueries <br><br> paris + italy - france → milan, london, venice |
| **50** | dog → cat, bird, horse <br><br> police → military, officers, guards <br><br> red → blue, yellow, green <br><br> tree → trees, fish, leaf <br><br> house → palace, court, castle | water + frozen → salt, dry, fish <br><br> king + woman - man → queen, prince, princess <br><br> plant + tall + wood → stone, fish, water <br><br> nature - inside → morality, zoology, temperament <br><br> paris + italy - france → venice, milan, berlin |
| **100** | dog → dogs, cat, horse <br><br> police → military, officers, officer <br><br> red → blue, yellow, green <br><br> tree → trees, flowers, garden <br><br> house → hall, room, houses | water + frozen → dry, fish, ice <br><br> king + woman - man → queen, princess, prince <br><br> plant + tall + wood → fish, plants, stone <br><br> nature - inside → morality, altruism, paradoxologia <br><br> paris + italy - france → milan, venice, rome |