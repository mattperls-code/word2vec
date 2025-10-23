#ifndef SRC_CBOW
#define SRC_CBOW

#include <string>
#include <vector>
#include <unordered_map>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/utils/xtensor_simd.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>

class CBOW_Partials
{
    private:
        bool empty;

    public:
        std::unordered_map<unsigned int, xt::xtensor<float, 1>> inputEmbedTable;
        xt::xtensor<float, 2> outputEmbedMatrix;

        CBOW_Partials(): empty(true), inputEmbedTable(), outputEmbedMatrix() {};
        CBOW_Partials(std::unordered_map<unsigned int, xt::xtensor<float, 1>> inputEmbedTable, xt::xtensor<float, 2> outputEmbedMatrix): empty(false), inputEmbedTable(inputEmbedTable), outputEmbedMatrix(outputEmbedMatrix) {};

        void operator+=(const CBOW_Partials& other);
};

class CBOW
{
    private:
        std::vector<unsigned int> corpus;
        std::unordered_map<std::string, unsigned int> vocabMapFromWord;
        std::vector<std::string> vocabMapFromIndex;

        int contextWindowSize;
        size_t embedDimensions;

        std::vector<xt::xtensor<float, 1>> inputEmbedTable;
        xt::xtensor<float, 2> outputEmbedMatrix;

        void assertWordInVocab(std::string word, std::string caller);

        CBOW_Partials calculateLossPartials(std::vector<unsigned int> context, unsigned int expectedWord);
        void applyLossPartials(CBOW_Partials partials, float scalar);

        xt::xtensor<float, 1> calculateFF(std::vector<std::string> context);
        std::vector<std::string> predictNextWords(std::vector<std::string> context, int n);

        float calculateLoss(std::vector<std::string> context, std::string expectedWord);

    public:
        CBOW(std::vector<std::string> corpus, int contextWindowSize, size_t embedDimensions);

        void print();

        void train(int batchSize, float learningRate);

        std::vector<std::string> findSimilar(std::string word, int n);

        bool save(std::string backupFilePath);
        bool load(std::string backupFilePath);

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->corpus, this->vocabMapFromWord, this->vocabMapFromIndex, this->contextWindowSize, this->embedDimensions, this->inputEmbedTable, this->outputEmbedMatrix);
        };
};

#endif