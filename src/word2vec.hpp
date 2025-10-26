#ifndef SRC_WORD2VEC
#define SRC_WORD2VEC

#include <string>
#include <vector>
#include <unordered_map>

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>

class Word2Vec
{
    private:
        std::unordered_map<std::string, unsigned int> vocabMapFromWord;
        std::vector<std::string> vocabMapFromIndex;

        std::size_t contextWindowSize;
        std::size_t negativeSampleCount;
        std::size_t embedDimensions;
        
        std::vector<float> inputEmbedMatrix;
        std::vector<float> outputEmbedMatrix;

        void assertWordInVocab(std::string word, std::string caller);

        void train(std::vector<unsigned int> context, unsigned int expectedWord, float learningRate);

    public:
        std::vector<unsigned int> corpus;

        Word2Vec() = default;

        Word2Vec(std::vector<std::string> corpus, std::size_t contextWindowSize, std::size_t negativeSampleCount, std::size_t embedDimensions);

        void trainStochasticEpoch(float learningRate);

        void postProcess();

        std::vector<float> getEmbedding(std::string word);

        std::vector<std::string> findSimilarToEmbedding(std::vector<float> embedding, int n);
        std::vector<std::string> findSimilarToWord(std::string word, int n);

        bool save(std::string backupFilePath);
        bool load(std::string backupFilePath);

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->corpus, this->vocabMapFromWord, this->vocabMapFromIndex, this->contextWindowSize, this->embedDimensions, this->inputEmbedMatrix, this->outputEmbedMatrix);
        };
};

#endif