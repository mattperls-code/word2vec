#include "word2vec.hpp"

#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <queue>
#include <filesystem>
#include <fstream>

#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/io/xio.hpp>

void Word2VecLossPartials::operator+=(const Word2VecLossPartials& other)
{
    if (this->empty) {
        this->empty = false;
        this->inputEmbedTable = other.inputEmbedTable;
        this->outputEmbedMatrix = other.outputEmbedMatrix;

        return;
    }
    else {
        for (const auto& [word, embedding] : other.inputEmbedTable) {
            if (this->inputEmbedTable.contains(word)) this->inputEmbedTable[word] += embedding;

            else this->inputEmbedTable[word] = embedding;
        }

        this->outputEmbedMatrix += other.outputEmbedMatrix;
    }
};

Word2Vec::Word2Vec(std::vector<std::string> corpus, int contextWindowSize, int negativeSampleCount, size_t embedDimensions)
{
    if (contextWindowSize < 1) throw std::runtime_error("Word2Vec constructor: contextWindowSize must be at least 1");

    this->contextWindowSize = contextWindowSize;

    if (negativeSampleCount < 1) throw std::runtime_error("Word2Vec constructor: negativeSampleCount must be at least 1");

    this->negativeSampleCount = negativeSampleCount;

    /* populate validated corpus and vocabulary */

    if (corpus.size() < 1 + 2 * this->contextWindowSize) throw std::runtime_error("Word2Vec constructor: invalid corpus"); // at least 5 before 5 after

    this->corpus.reserve(corpus.size());

    // estimate vocabulary size using heap's law: V ~ K • N^B
    // approximate as K = 50, N = 10C, B = 0.5
    this->vocabMapFromIndex.reserve(50 * sqrt(10 * corpus.size()));
    this->vocabMapFromWord.reserve(50 * sqrt(10 * corpus.size()));

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

    this->inputEmbedTable.reserve(this->vocabMapFromIndex.size());
    
    for (const auto& _ : this->vocabMapFromIndex) this->inputEmbedTable.push_back(xt::random::rand<float>({ this->embedDimensions }) - 0.5);

    this->outputEmbedMatrix = xt::random::rand<float>({ this->vocabMapFromIndex.size(), this->embedDimensions }) - 0.5;
};

void Word2Vec::assertWordInVocab(std::string word, std::string caller)
{
    if (!this->vocabMapFromWord.contains(word)) throw std::runtime_error("Word2Vec " + caller + ": word \"" + word + "\" is not in vocab");
};

Word2VecLossPartials Word2Vec::calculateSoftmaxLossPartials(std::vector<unsigned int> context, unsigned int expectedWord)
{
    // FEEDFORWARD

    /* projection is avg of context embeddings  */

    xt::xtensor<float, 1> projection = xt::zeros<float>({ this->embedDimensions });

    for (int i = 0;i<context.size();i++) projection += this->inputEmbedTable[context[i]];

    projection /= context.size();

    /* apply output embed matrix transform */

    xt::xtensor<float, 1> prenormalizedOutput = xt::linalg::dot(this->outputEmbedMatrix, projection);

    /* softmax */

    xt::xtensor<float, 1> expPrenormalizedOutput = exp(prenormalizedOutput - xt::amax(prenormalizedOutput)());

    xt::xtensor<float, 1> normalizedOutput = expPrenormalizedOutput / sum(expPrenormalizedOutput);

    // BACKPROP

    /* simplified cce derivative using softmax output */

    xt::xtensor<float, 1> dLossWrtPrenormalizedOutput = normalizedOutput;
    dLossWrtPrenormalizedOutput(expectedWord) -= 1.0;

    /* chain rule to determine loss partial wrt output embed matrix and context embed */

    xt::xtensor<float, 2> dLossWrtOutputEmbedMatrix = xt::linalg::outer(dLossWrtPrenormalizedOutput, projection);

    xt::xtensor<float, 1> dLossWrtContextEmbed = xt::linalg::dot(xt::transpose(this->outputEmbedMatrix), dLossWrtPrenormalizedOutput) / context.size();
    
    std::unordered_map<unsigned int, xt::xtensor<float, 1>> dLossWrtInputEmbedTable;

    for (const auto& word : context) dLossWrtInputEmbedTable[word] = dLossWrtContextEmbed;

    return Word2VecLossPartials(dLossWrtInputEmbedTable, dLossWrtOutputEmbedMatrix);
};

Word2VecLossPartials Word2Vec::calculateNegativeSamplingLossPartials(std::vector<unsigned int> context, unsigned int expectedWord)
{
    // FEEDFORWARD

    /* projection is avg of context embeddings  */

    xt::xtensor<float, 1> projection = xt::zeros<float>({ this->embedDimensions });

    for (int i = 0;i<context.size();i++) projection += this->inputEmbedTable[context[i]];

    projection /= context.size();

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<unsigned int> getRandomWord(0, this->vocabMapFromIndex.size() - 1);

    std::vector<unsigned int> negativeSamples;
    negativeSamples.reserve(this->negativeSampleCount);

    for (int i = 0;i<this->negativeSampleCount;i++) {
        unsigned int negativeSample =  getRandomWord(gen);

        if (negativeSample != expectedWord) negativeSamples.push_back(negativeSample);
    }

    xt::xtensor<float, 2> dLossWrtOutputEmbedMatrix = xt::zeros_like(this->outputEmbedMatrix);
    xt::xtensor<float, 1> dLossWrtInputContextEmbed = xt::zeros<float>({ this->embedDimensions });

    /* gradients for expected word */

    const xt::xtensor<float, 1>& targetWordOutputEmbedding = xt::view(this->outputEmbedMatrix, expectedWord, xt::all());

    float expectedWordScore = xt::linalg::dot(projection, targetWordOutputEmbedding)();
    float dLossWrtExpectedWordScore = 1.0 / (1.0 + std::exp(-expectedWordScore)) - 1.0;

    xt::view(dLossWrtOutputEmbedMatrix, expectedWord, xt::all()) += dLossWrtExpectedWordScore * projection;
    dLossWrtInputContextEmbed += dLossWrtExpectedWordScore * targetWordOutputEmbedding;

    /* gradients for negative samples */

    for (unsigned int negativeSample : negativeSamples) {
        const xt::xtensor<float, 1>& negativeSampleOutputEmbedding = xt::view(this->outputEmbedMatrix, negativeSample, xt::all());

        float negativeSampleScore = xt::linalg::dot(projection, negativeSampleOutputEmbedding)();
        float dLossWrtNegativeSampleScore = 1.0 / (1.0 + std::exp(-negativeSampleScore));

        xt::view(dLossWrtOutputEmbedMatrix, negativeSample, xt::all()) += dLossWrtNegativeSampleScore * projection;
        dLossWrtInputContextEmbed += dLossWrtNegativeSampleScore * negativeSampleOutputEmbedding;
    }

    dLossWrtInputContextEmbed /= context.size();

    std::unordered_map<unsigned int, xt::xtensor<float, 1>> dLossWrtInputEmbedTable;

    for (unsigned int word : context) dLossWrtInputEmbedTable[word] = dLossWrtInputContextEmbed;

    return Word2VecLossPartials(dLossWrtInputEmbedTable, dLossWrtOutputEmbedMatrix);
};

void Word2Vec::applyLossPartials(Word2VecLossPartials partials, float scalar)
{
    for (const auto& [word, dLossWrtEmbed] : partials.inputEmbedTable) this->inputEmbedTable[word] -= scalar * dLossWrtEmbed;

    this->outputEmbedMatrix -= partials.outputEmbedMatrix * scalar;
};

void Word2Vec::trainRandomBatch(int batchSize, float learningRate)
{
    Word2VecLossPartials batchLossPartials;

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<size_t> getRandomCorpusIndex(0, this->corpus.size() - 1 - 2 * this->contextWindowSize);

    for (int i = 0;i<batchSize;i++) {
        int randomCorpusIndex = this->contextWindowSize + getRandomCorpusIndex(gen);

        std::vector<unsigned int> contextWindow;
        contextWindow.reserve(2 * this->contextWindowSize);

        for (int j = 1;j<=this->contextWindowSize;j++) {
            contextWindow.push_back(this->corpus[randomCorpusIndex + j]);
            contextWindow.push_back(this->corpus[randomCorpusIndex - j]);
        }

        // batchLossPartials += this->calculateSoftmaxLossPartials(contextWindow, this->corpus[randomCorpusIndex]);
        batchLossPartials += this->calculateNegativeSamplingLossPartials(contextWindow, this->corpus[randomCorpusIndex]);
    }

    this->applyLossPartials(batchLossPartials, learningRate / batchSize);
};

void Word2Vec::trainStochasticEpoch(float learningRate)
{
    std::vector<int> indices(this->corpus.size());
    std::iota(indices.begin(), indices.end(), 0);

    static std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    std::cout << "Stochastic Training Epoch Started" << std::endl;

    int count = 0;

    for (int index : indices) {
        if (count % 1000 == 0) std::cout << "Training " << std::fixed << std::setprecision(3) << (100.0f * (float) count / (float) this->corpus.size()) << "\% finished" << std::endl;

        std::vector<unsigned int> contextWindow;
        contextWindow.reserve(2 * this->contextWindowSize);

        for (int i = 1;i<=this->contextWindowSize;i++) {
            if (index - i >= 0) contextWindow.push_back(this->corpus[index - i]);
            if (index + i < this->corpus.size()) contextWindow.push_back(this->corpus[index + i]);
        }

        // this->applyLossPartials(this->calculateSoftmaxLossPartials(contextWindow, this->corpus[index]), learningRate);
        this->applyLossPartials(this->calculateNegativeSamplingLossPartials(contextWindow, this->corpus[index]), learningRate);

        count++;
    }
};

std::vector<std::string> Word2Vec::findSimilar(std::string word, int n)
{
    this->assertWordInVocab(word, "findSimilar");

    if (n < 1) throw std::runtime_error("Word2Vec findSimilar: n must be at least 1");

    // pairs are { (negative for min-heap) similarity, wordIndex }, pq auto compares by first item
    std::priority_queue<std::pair<float, unsigned int>> mostSimilar;

    unsigned int wordIndex = this->vocabMapFromWord[word];

    xt::xtensor<float, 1> wordEmbedding = this->inputEmbedTable[wordIndex];

    for (int i = 0;i<this->inputEmbedTable.size();i++) {
        if (i == wordIndex) continue;

        float similarity = xt::linalg::dot(wordEmbedding, this->inputEmbedTable[i])() / xt::linalg::norm(wordEmbedding) / xt::linalg::norm(this->inputEmbedTable[i]);

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

namespace cereal {
    template <class Archive, size_t N>
    void save(Archive& ar, const xt::xtensor<float, N>& vec)
    {
        std::vector<std::size_t> shape(vec.dimension());
        for (size_t i = 0;i<vec.dimension();i++) shape[i] = vec.shape()[i];

        std::vector<float> data(vec.begin(), vec.end());

        ar(shape, data);
    };

    template <class Archive, size_t N>
    void load(Archive& ar, xt::xtensor<float, N>& vec)
    {
        std::vector<size_t> shape;
        std::vector<float> data;

        ar(shape, data);

        vec = xt::adapt(data, shape);
    };
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