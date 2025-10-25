#include "word2vec.hpp"

#include <fstream>
#include <stdexcept>

#include <xtensor/io/xio.hpp>

Word2Vec model;

void initModel(std::string corpusFilePath)
{
    std::ifstream corpusFile(corpusFilePath);

    if (!corpusFile) throw std::runtime_error("Failed to open corpus");

    std::vector<std::string> corpus;
    corpus.reserve(2000000);

    std::string word;
    while (corpusFile >> word) {
        corpus.push_back(word);

        // if (corpus.size() == 2000000) break;
    }

    corpusFile.close();

    std::cout << "Corpus size: " << corpus.size() << std::endl;

    Word2Vec model(corpus, 4, 10, 160);

    std::cout << "Initialized Model" << std::endl << std::endl;
};

void loadModel()
{
    std::cout << "Saved Model File Path? ";

    std::string savedModelFilePath;

    std::cin >> savedModelFilePath;

    std::cout << std::endl;

    bool loaded = model.load(savedModelFilePath);

    if (loaded) std::cout << "Successfully Loaded";

    else std::cout << "An Error Occurred";

    std::cout << std::endl << std::endl;
};

void saveModel()
{
    std::cout << "Saved Model File Path? ";

    std::string savedModelFilePath;

    std::cin >> savedModelFilePath;

    std::cout << std::endl;

    bool saved = model.save(savedModelFilePath);

    if (saved) std::cout << "Successfully Saved";

    else std::cout << "An Error Occurred";

    std::cout << std::endl << std::endl;
};

void trainModel()
{
    std::cout << "Starting Training Epoch" << std::endl;

    model.trainStochasticEpoch(0.1);

    std::cout << "Finished Training Epoch" << std::endl << std::endl;
};

void testModel()
{
    std::cout << "Similar Embeddings" << std::endl;

    std::vector<std::string> words = { "cat", "dog", "king", "queen", "black", "white", "tree", "house" };

    for (const auto& word : words) {
        std::cout << word << ": ";

        try {
            std::vector<std::string> similarToWord = model.findSimilar(word, 8);

            for (const auto& similarWord : similarToWord) std::cout << similarWord << " ";

            std::cout << std::endl;
        } catch (std::runtime_error e) {
            std::cout << word << " failed: " << e.what() << std::endl;
        };
    }

    std::cout << std::endl;
};

void getSimilar()
{
    std::cout << "Target Word? ";

    std::string word;

    std::cin >> word;

    std::cout << std::endl;
    
    try {
        std::vector<std::string> similarToWord = model.findSimilar(word, 8);

        for (const auto& similarWord : similarToWord) std::cout << similarWord << " ";

        std::cout << std::endl;
    } catch (std::runtime_error e) {
        std::cout << word << " failed: " << e.what() << std::endl;
    };

    std::cout << std::endl;
};

int main()
{
    initModel("./app/cleanCorpus/text8");
    
    std::cout << std::endl << "Welcome to the Word2Vec demo!" << std::endl << std::endl;

    while (true) {
        std::string command;

        std::cout << "Enter a command (LOAD, SAVE, TRAIN, TEST, SIMILAR, EXIT): ";

        std::cin >> command;

        std::cout << std::endl;

        if (command == "LOAD") loadModel();

        else if (command == "SAVE") saveModel();

        else if (command == "TRAIN") trainModel();

        else if (command == "TEST") testModel();

        else if (command == "SIMILAR") getSimilar();

        else if (command == "EXIT") break;

        else std::cout << "Command \"" << command << "\" is unrecognized." << std::endl << std::endl;
    }

    return 0;
}