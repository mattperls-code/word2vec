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
    corpus.reserve(80000);

    std::string word;
    while (corpusFile >> word) {
        corpus.push_back(word);

        if (corpus.size() == 80000) break;
    }

    corpusFile.close();

    std::cout << "Corpus size: " << corpus.size() << std::endl;

    Word2Vec model(corpus, 5, 160);

    std::cout << "Built model" << std::endl << std::endl;

    for (int i = 0;i<20*80000/20;i++) {
        if (i % 10 == 0) std::cout << "Batch " << i << std::endl << std::endl;

        model.train(20, 10.0);

        if (i % 10 == 0) printSimilarEmbeddings(model);

        if (i % 500 == 0) model.save("./results/backups/afterBatch" + std::to_string(i));
    }

    return 0;
}

/*

    Similar Embeddings (Batch 14730)

    cat: bird woman moment mice swallow tapers child made 
    dog: clerk linnet princess maiden child swineherd prince person 
    king: miller well royal snow some broken moon floor 
    queen: flakes woman palace green swallow mirror evening sledge 
    black: earth answered by street sky low clerk life 
    white: sweet our sang room kay kind with act 
    tree: councillor giant emperor swallow people man dragged river 
    house: court large empty councillor children emperor hall than

*/