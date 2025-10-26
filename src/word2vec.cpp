#include "word2vec.hpp"

#include <iostream>
#include <span>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <queue>
#include <filesystem>
#include <fstream>

Word2Vec::Word2Vec(std::vector<std::string> corpus, std::size_t contextWindowSize, std::size_t negativeSampleCount, std::size_t embedDimensions)
{
    if (contextWindowSize == 0) throw std::runtime_error("Word2Vec constructor: invalid context window size");

    this->contextWindowSize = contextWindowSize;

    if (negativeSampleCount == 0) throw std::runtime_error("Word2Vec constructor: invalid negative sample count");

    this->negativeSampleCount = negativeSampleCount;

    /* populate validated corpus and vocabulary */

    if (corpus.size() < 1 + 2 * this->contextWindowSize) throw std::runtime_error("Word2Vec constructor: invalid corpus"); // at least 5 before 5 after

    this->corpus.reserve(corpus.size());

    // estimate vocabulary size using heap's law: V ~ K • N^B
    // approximate as K = 50, N = |corpus|, B = 0.4
    int approxVocabSize = 50.0 * pow(corpus.size(), 0.4);

    this->vocabMapFromIndex.reserve(approxVocabSize);
    this->vocabMapFromWord.reserve(approxVocabSize);

    for (const auto& word : corpus) {
        if (!vocabMapFromWord.contains(word)) {
            this->vocabMapFromWord[word] = this->vocabMapFromIndex.size();
            this->vocabMapFromIndex.push_back(word);
        }

        this->corpus.push_back(this->vocabMapFromWord[word]);
    }

    /* random init embeddings */

    if (embedDimensions == 0) throw std::runtime_error("Word2Vec constructor: invalid embed dimensions");

    this->embedDimensions = embedDimensions;

    std::size_t embedMatrixSize = this->vocabMapFromIndex.size() * this->embedDimensions;

    this->inputEmbedMatrix.reserve(embedMatrixSize);
    this->outputEmbedMatrix.reserve(embedMatrixSize);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> getRandomWeight(-0.1, 0.1);

    for (int i = 0;i<embedMatrixSize;i++) {
        this->inputEmbedMatrix.push_back(getRandomWeight(gen));
        this->outputEmbedMatrix.push_back(getRandomWeight(gen));
    };
};

void Word2Vec::assertWordInVocab(std::string word, std::string caller)
{
    if (!this->vocabMapFromWord.contains(word)) throw std::runtime_error("Word2Vec " + caller + ": word \"" + word + "\" is not in vocab");
};

void Word2Vec::train(std::vector<unsigned int> context, unsigned int expectedWord, float learningRate)
{
    // FEEDFORWARD

    /* projection is avg of context embeddings  */

    std::vector<float> projection(this->embedDimensions, 0.0);

    for (unsigned int word : context) 
        for (int i = 0;i<this->embedDimensions;i++) projection[i] += this->inputEmbedMatrix[word * this->embedDimensions + i];
        
    for (float& component : projection) component /= context.size();

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned int> getRandomWord(0, this->vocabMapFromIndex.size() - 1);

    std::vector<unsigned int> negativeSamples;
    negativeSamples.reserve(this->negativeSampleCount);

    for (int i = 0;i<this->negativeSampleCount;i++) {
        unsigned int negativeSample =  getRandomWord(gen);

        if (negativeSample != expectedWord) negativeSamples.push_back(negativeSample);
    }

    std::vector<float> dLossWrtInputContextEmbed(this->embedDimensions, 0.0);

    /* gradients for expected word */

    float expectedWordScore = 0.0;
    for (int i = 0;i<this->embedDimensions;i++) expectedWordScore += projection[i] * this->outputEmbedMatrix[expectedWord * this->embedDimensions + i];

    float dLossWrtExpectedWordScore = 1.0 / (1.0 + exp(-expectedWordScore)) - 1.0;

    for (int i = 0;i<this->embedDimensions;i++) dLossWrtInputContextEmbed[i] += dLossWrtExpectedWordScore * this->outputEmbedMatrix[expectedWord * this->embedDimensions + i];
    for (int i = 0;i<this->embedDimensions;i++) this->outputEmbedMatrix[expectedWord * this->embedDimensions + i] -= learningRate * dLossWrtExpectedWordScore * projection[i];

    /* gradients for negative samples */

    for (unsigned int negativeSample : negativeSamples) {
        float negativeSampleScore = 0.0;
        for (int i = 0;i<this->embedDimensions;i++) negativeSampleScore += projection[i] * this->outputEmbedMatrix[negativeSample * this->embedDimensions + i];

        float dLossWrtNegativeSampleScore = 1.0 / (1.0 + std::exp(-negativeSampleScore));

        for (int i = 0;i<this->embedDimensions;i++) dLossWrtInputContextEmbed[i] += dLossWrtNegativeSampleScore * this->outputEmbedMatrix[negativeSample * this->embedDimensions + i];
        for (int i = 0;i<this->embedDimensions;i++) this->outputEmbedMatrix[negativeSample * this->embedDimensions + i] -= learningRate * dLossWrtNegativeSampleScore * projection[i];
    }

    for (float& component : dLossWrtInputContextEmbed) component /= context.size();

    for (unsigned int word : context)
        for (int i = 0;i<this->embedDimensions;i++) this->inputEmbedMatrix[word * this->embedDimensions + i] -= learningRate * dLossWrtInputContextEmbed[i];
};

void Word2Vec::trainStochasticEpoch(float learningRate)
{
    std::vector<int> indices(this->corpus.size());
    std::iota(indices.begin(), indices.end(), 0);

    static std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (int index : indices) {
        std::vector<unsigned int> contextWindow;
        contextWindow.reserve(2 * this->contextWindowSize);

        for (int i = 1;i<=this->contextWindowSize;i++) {
            if (index - i >= 0) contextWindow.push_back(this->corpus[index - i]);
            if (index + i < this->corpus.size()) contextWindow.push_back(this->corpus[index + i]);
        }

        this->train(contextWindow, this->corpus[index], learningRate);
    }

    std::cout << "Stochastic Training Epoch Finished" << std::endl;
};

void Word2Vec::postProcess()
{
    /* shift by embedding mean and then normalize */

    std::vector<float> meanInputEmbed(this->embedDimensions, 0.0);
    std::vector<float> meanOutputEmbed(this->embedDimensions, 0.0);

    for (int i = 0;i<this->vocabMapFromIndex.size();i++) {
        for (int j = 0;j<this->embedDimensions;j++) {
            meanInputEmbed[j] += this->inputEmbedMatrix[i * this->embedDimensions + j];
            meanOutputEmbed[j] += this->outputEmbedMatrix[i * this->embedDimensions + j];
        }
    }

    for (int i = 0;i<this->embedDimensions;i++) {
        meanInputEmbed[i] /= this->vocabMapFromIndex.size();
        meanOutputEmbed[i] /= this->vocabMapFromIndex.size();
    }
    
    for (int i = 0;i<this->vocabMapFromIndex.size();i++) {
        float inputEmbedMagnitude = 0.0;
        float outputEmbedMagnitude = 0.0;

        for (int j = 0;j<this->embedDimensions;j++) {
            this->inputEmbedMatrix[i * this->embedDimensions + j] -= meanInputEmbed[j];
            this->outputEmbedMatrix[i * this->embedDimensions + j] -= meanOutputEmbed[j];

            inputEmbedMagnitude += this->inputEmbedMatrix[i * this->embedDimensions + j] * this->inputEmbedMatrix[i * this->embedDimensions + j];
            outputEmbedMagnitude += this->outputEmbedMatrix[i * this->embedDimensions + j] * this->outputEmbedMatrix[i * this->embedDimensions + j];
        }

        inputEmbedMagnitude = sqrt(inputEmbedMagnitude);
        outputEmbedMagnitude = sqrt(outputEmbedMagnitude);

        for (int j = 0;j<this->embedDimensions;j++) {
            this->inputEmbedMatrix[i * this->embedDimensions + j] /= inputEmbedMagnitude;
            this->outputEmbedMatrix[i * this->embedDimensions + j] /= outputEmbedMagnitude;
        }
    }
};

std::vector<float> Word2Vec::getEmbedding(std::string word)
{
    this->assertWordInVocab(word, "getEmbedding");

    unsigned int wordIndex = this->vocabMapFromWord[word];

    return std::vector<float>(
        this->outputEmbedMatrix.begin() + wordIndex * this->embedDimensions,
        this->outputEmbedMatrix.begin() + (wordIndex + 1) * this->embedDimensions
    );
};

std::vector<std::string> Word2Vec::findSimilarToEmbedding(std::vector<float> embedding, int n)
{
    if (embedding.size() != this->embedDimensions) throw std::runtime_error("Word2Vec findSimilarToEmbedding: embedding is the wrong size");

    if (n < 1) throw std::runtime_error("Word2Vec findSimilarToEmbedding: n must be at least 1");

    // pairs are { (negative for min-heap) similarity, wordIndex }, pq auto compares by first item
    std::priority_queue<std::pair<float, unsigned int>> mostSimilar;

    for (int i = 0;i<this->vocabMapFromIndex.size();i++) {
        float similarity = 0.0;
        float otherMagnitude = 0.0;
        for (int j = 0;j<this->embedDimensions;j++) {
            similarity += embedding[j] * this->outputEmbedMatrix[i * this->embedDimensions + j];
            otherMagnitude += this->outputEmbedMatrix[i * this->embedDimensions + j] * this->outputEmbedMatrix[i * this->embedDimensions + j];
        }

        similarity /= sqrt(otherMagnitude);

        mostSimilar.push({ -similarity, i });

        if (mostSimilar.size() > n) mostSimilar.pop();
    }

    std::vector<std::string> topN;

    while (!mostSimilar.empty()) {
        topN.push_back(this->vocabMapFromIndex[mostSimilar.top().second]);

        mostSimilar.pop();
    }

    std::reverse(topN.begin(), topN.end());

    return topN;
};

std::vector<std::string> Word2Vec::findSimilarToWord(std::string word, int n)
{
    this->assertWordInVocab(word, "findSimilarToWord");

    if (n < 1) throw std::runtime_error("Word2Vec findSimilarToWord: n must be at least 1");

    // pairs are { (negative for min-heap) similarity, wordIndex }, pq auto compares by first item
    std::priority_queue<std::pair<float, unsigned int>> mostSimilar;

    unsigned int wordIndex = this->vocabMapFromWord[word];

    for (int i = 0;i<this->vocabMapFromIndex.size();i++) {
        if (i == wordIndex) continue;

        float similarity = 0.0;
        for (int j = 0;j<this->embedDimensions;j++) similarity += this->outputEmbedMatrix[wordIndex * this->embedDimensions + j] * this->outputEmbedMatrix[i * this->embedDimensions + j];

        mostSimilar.push({ -similarity, i });

        if (mostSimilar.size() > n) mostSimilar.pop();
    }

    std::vector<std::string> topN;

    while (!mostSimilar.empty()) {
        topN.push_back(this->vocabMapFromIndex[mostSimilar.top().second]);

        mostSimilar.pop();
    }

    std::reverse(topN.begin(), topN.end());

    return topN;
};

bool Word2Vec::save(std::string backupFilePath)
{
    try {
        std::filesystem::path path(backupFilePath);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());

        std::ofstream outputStream(backupFilePath);

        cereal::BinaryOutputArchive archive(outputStream);

        archive(*this);

        return true;
    } catch (const std::exception&) {
        return false;
    }
};

bool Word2Vec::load(std::string backupFilePath)
{
    try {
        std::filesystem::path path(backupFilePath);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());

        std::ifstream inputStream(backupFilePath);

        cereal::BinaryInputArchive archive(inputStream);

        archive(*this);

        return true;
    } catch (const std::exception&) {
        return false;
    }
};