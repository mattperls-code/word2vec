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
    while (corpusFile >> word) corpus.push_back(word);

    corpusFile.close();

    std::cout << "Corpus size: " << corpus.size() << std::endl;

    model = Word2Vec(corpus, 4, 10, 150);

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
    std::cout << "How Many Epochs? ";

    unsigned int n;

    std::cin >> n;

    std::cout << std::endl;

    for (int i = 0;i<n;i++) model.trainStochasticEpoch(0.02);

    std::cout << "Finished Training" << std::endl << std::endl;
};

void postProcess()
{
    std::cout << "Post Processing" << std::endl;

    model.postProcess();

    std::cout << std::endl;
};

void testModel()
{
    std::cout << "Similar Embeddings" << std::endl;

    std::vector<std::string> words = { "cat", "dog", "king", "queen", "black", "white", "tree", "house" };

    for (const auto& word : words) {
        std::cout << word << ": ";

        try {
            std::vector<std::string> similarToWord = model.findSimilarToWord(word, 8);

            for (const auto& similarWord : similarToWord) std::cout << similarWord << " ";

            std::cout << std::endl;
        } catch (std::runtime_error err) {
            std::cout << word << " failed: " << err.what() << std::endl;
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
        std::vector<std::string> similarToWord = model.findSimilarToWord(word, 8);

        for (const auto& similarWord : similarToWord) std::cout << similarWord << " ";

        std::cout << std::endl;
    } catch (std::runtime_error err) {
        std::cout << word << " failed: " << err.what() << std::endl;
    };

    std::cout << std::endl;
};

void evaluate()
{
    std::cout << "Evaluating Composition" << std::endl;

    std::vector<float> kingEmbedding = model.getEmbedding("king");
    std::vector<float> manEmbedding = model.getEmbedding("man");
    std::vector<float> womanEmbedding = model.getEmbedding("woman");

    std::vector<float> composition;

    for (int i = 0;i<kingEmbedding.size();i++) composition.push_back(kingEmbedding[i] + womanEmbedding[i] - manEmbedding[i]);

    std::vector<std::string> similarToComposition = model.findSimilarToEmbedding(composition, 8);

    std::cout << "Similar to (king + woman - man): ";

    for (const auto& similarWord : similarToComposition) std::cout << similarWord << " ";

    std::cout << std::endl << std::endl;
};

int main()
{
    std::cout << std::endl << "Welcome to the Word2Vec demo!" << std::endl << std::endl;

    initModel("./app/cleanCorpus/text8");

    while (true) {
        std::string command;

        std::cout << "Enter a command (LOAD, SAVE, TRAIN, POSTPROCESS, TEST, SIMILAR, EVALUATE, EXIT): ";

        std::cin >> command;

        std::cout << std::endl;

        if (command == "LOAD") loadModel();

        else if (command == "SAVE") saveModel();

        else if (command == "TRAIN") trainModel();

        else if (command == "POSTPROCESS") postProcess();

        else if (command == "TEST") testModel();

        else if (command == "SIMILAR") getSimilar();

        else if (command == "EVALUATE") evaluate();

        else if (command == "EXIT") break;

        else std::cout << "Command \"" << command << "\" is unrecognized." << std::endl << std::endl;
    }

    return 0;
}