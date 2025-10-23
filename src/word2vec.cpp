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

Word2Vec::Word2Vec(std::vector<std::string> corpus, int contextWindowSize, size_t embedDimensions)
{
    if (contextWindowSize < 1) throw std::runtime_error("Word2Vec constructor: contextWindowSize must be at least 1");

    this->contextWindowSize = contextWindowSize;

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

void Word2Vec::print()
{
    std::cout << "Embed Dimension: " << this->embedDimensions << std::endl << std::endl;

    std::cout << "Corpus: " << std::endl;

    for (auto word : this->corpus) std::cout << "\t" << word << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "VocabMapFromWord: " << std::endl;

    for (const auto [word, index] : this->vocabMapFromWord) std::cout << "\t" << word << ": " << index << std::endl;

    std::cout << std::endl;

    std::cout << "VocabMapFromIndex: " << std::endl;
    
    for (int i = 0;i<this->vocabMapFromIndex.size();i++) std::cout << "\t" << i << ": " << this->vocabMapFromIndex[i] << std::endl;

    std::cout << std::endl;

    std::cout << "InputEmbedTable: " << std::endl;
    
    for (int i = 0;i<this->inputEmbedTable.size();i++) std::cout << "\t" << this->vocabMapFromIndex[i] << ": " << this->inputEmbedTable[i] << std::endl;

    std::cout << std::endl;

    std::cout << "OutputEmbedTable: " << std::endl << this->outputEmbedMatrix << std::endl;
};

xt::xtensor<float, 1> Word2Vec::calculateFF(std::vector<std::string> context)
{
    for (const auto& word : context) this->assertWordInVocab(word, "calculateFF");

    /* projection is avg of context embeddings  */

    xt::xtensor<float, 1> projection = xt::zeros<float>({ this->embedDimensions });

    for (int i = 0;i<context.size();i++) projection += this->inputEmbedTable[this->vocabMapFromWord[context[i]]];

    projection /= context.size();

    /* apply output embed matrix transform */

    xt::xtensor<float, 1> prenormalizedOutput = xt::linalg::dot(this->outputEmbedMatrix, projection);

    /* softmax */

    xt::xtensor<float, 1> expPrenormalizedOutput = exp(prenormalizedOutput - xt::amax(prenormalizedOutput)());

    xt::xtensor<float, 1> normalizedOutput = expPrenormalizedOutput / sum(expPrenormalizedOutput);

    return normalizedOutput;
};

std::vector<std::string> Word2Vec::predictNextWords(std::vector<std::string> context, int n)
{
    for (const auto& word : context) this->assertWordInVocab(word, "predictNextWord");

    if (n >= this->vocabMapFromIndex.size()) n = this->vocabMapFromIndex.size() - 1;

    /* pull word from highest index of softmax output */

    xt::xtensor<float, 1> ffOutput = this->calculateFF(context);

    std::vector<std::string> topN;

    for (int i = 0;i<n;i++) {
        int maxIndex = xt::argmax(ffOutput)();

        topN.push_back(this->vocabMapFromIndex[maxIndex]);

        ffOutput(maxIndex) = 0.0;
    }

    return topN;
};

float Word2Vec::calculateLoss(std::vector<std::string> context, std::string expectedWord)
{
    for (const auto& word : context) this->assertWordInVocab(word, "calculateLoss");
    this->assertWordInVocab(expectedWord, "calculateLoss");

    xt::xtensor<float, 1> observed = this->calculateFF(context);

    /* categorical cross entropy loss. since expected output is a single one hot encoded, we can just use that */

    float epsilon = 1e-8;

    return -log(std::clamp(observed(this->vocabMapFromWord[expectedWord]), epsilon, 1.0f - epsilon));
};

Word2VecLossPartials Word2Vec::calculateLossPartials(std::vector<unsigned int> context, unsigned int expectedWord)
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

void Word2Vec::applyLossPartials(Word2VecLossPartials partials, float scalar)
{
    for (const auto& [word, dLossWrtEmbed] : partials.inputEmbedTable) this->inputEmbedTable[word] -= scalar * dLossWrtEmbed;

    this->outputEmbedMatrix -= partials.outputEmbedMatrix * scalar;
};

void Word2Vec::train(int batchSize, float learningRate)
{
    Word2VecLossPartials batchLossPartials;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<size_t> getRandomCorpusIndex(0, this->corpus.size() - 1 - 2 * this->contextWindowSize);

    for (int i = 0;i<batchSize;i++) {
        int randomCorpusIndex = this->contextWindowSize + getRandomCorpusIndex(gen);

        std::vector<unsigned int> contextWindow;

        for (int j = 1;j<=this->contextWindowSize;j++) {
            contextWindow.push_back(this->corpus[randomCorpusIndex + j]);
            contextWindow.push_back(this->corpus[randomCorpusIndex - j]);
        }

        batchLossPartials += this->calculateLossPartials(contextWindow, this->corpus[randomCorpusIndex]);
    }

    this->applyLossPartials(batchLossPartials, learningRate / batchSize);
};

std::vector<std::string> Word2Vec::findSimilar(std::string word, int n)
{
    this->assertWordInVocab(word, "findSimilar");

    if (n < 1) throw std::runtime_error("Word2Vec findSimilar: n must be positive");

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