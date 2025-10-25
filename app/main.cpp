#include "word2vec.hpp"

#include <fstream>
#include <stdexcept>

#include <xtensor/io/xio.hpp>

void printSimilarEmbeddings(Word2Vec& model)
{
    std::cout << "Similar Embeddings" << std::endl;

    std::vector<std::string> words = { "cat", "dog", "king", "queen", "black", "white", "tree", "house" };

    for (const auto& word : words) {
        std::cout << word << ": ";

        std::vector<std::string> similarToWord = model.findSimilar(word, 8);

        for (const auto& similarWord : similarToWord) std::cout << similarWord << " ";

        std::cout << std::endl;
    }

    std::cout << std::endl;
};

int main()
{
    std::ifstream corpusFile("./app/cleanCorpus/fairy_tales.txt");

    if (!corpusFile) throw std::runtime_error("Failed to open corpus");

    std::vector<std::string> corpus;
    corpus.reserve(300000);

    std::string word;
    while (corpusFile >> word) {
        corpus.push_back(word);

        if (corpus.size() == 300000) break;
    }

    corpusFile.close();

    std::cout << "Corpus size: " << corpus.size() << std::endl;

    Word2Vec model(corpus, 4, 10, 160);

    std::cout << "Built model" << std::endl << std::endl;

    printSimilarEmbeddings(model);

    model.save("./results/backups/afterEpoch0");

    for (int i = 0;i<200;i++) {
        model.trainStochasticEpoch(0.1);

        printSimilarEmbeddings(model);

        model.save("./results/backups/afterEpoch" + std::to_string(i + 1));
    }

    return 0;
}