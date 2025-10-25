#ifndef SRC_WORD2VEC
#define SRC_WORD2VEC

#include <string>
#include <vector>
#include <unordered_map>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/utils/xtensor_simd.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>

class Word2VecLossPartials
{
    private:
        bool empty;

    public:
        std::unordered_map<unsigned int, xt::xtensor<float, 1>> inputEmbedTable;
        xt::xtensor<float, 2> outputEmbedMatrix;

        Word2VecLossPartials(): empty(true), inputEmbedTable(), outputEmbedMatrix() {};
        Word2VecLossPartials(std::unordered_map<unsigned int, xt::xtensor<float, 1>> inputEmbedTable, xt::xtensor<float, 2> outputEmbedMatrix): empty(false), inputEmbedTable(inputEmbedTable), outputEmbedMatrix(outputEmbedMatrix) {};

        void operator+=(const Word2VecLossPartials& other);
};

class Word2Vec
{
    private:
        std::vector<unsigned int> corpus;
        std::unordered_map<std::string, unsigned int> vocabMapFromWord;
        std::vector<std::string> vocabMapFromIndex;

        int contextWindowSize;
        int negativeSampleCount;
        size_t embedDimensions;

        std::vector<xt::xtensor<float, 1>> inputEmbedTable;
        xt::xtensor<float, 2> outputEmbedMatrix;

        void assertWordInVocab(std::string word, std::string caller);

        Word2VecLossPartials calculateSoftmaxLossPartials(std::vector<unsigned int> context, unsigned int expectedWord);
        Word2VecLossPartials calculateNegativeSamplingLossPartials(std::vector<unsigned int> context, unsigned int expectedWord);
        void applyLossPartials(Word2VecLossPartials partials, float scalar);

    public:
        Word2Vec() = default;

        Word2Vec(std::vector<std::string> corpus, int contextWindowSize, int negativeSampleCount, size_t embedDimensions);

        void trainRandomBatch(int batchSize, float learningRate);
        void trainStochasticEpoch(float learningRate);

        std::vector<std::string> findSimilar(std::string word, int n);

        bool save(std::string backupFilePath);
        bool load(std::string backupFilePath);

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->corpus, this->vocabMapFromWord, this->vocabMapFromIndex, this->contextWindowSize, this->embedDimensions, this->inputEmbedTable, this->outputEmbedMatrix);
        };
};

#endif