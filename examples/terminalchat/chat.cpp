#include "rwkv.h"
#include <filesystem>

int main(){
    std::string chatRecord = "The following is a coherent verbose detailed conversation between a girl named Alice and her friend Bob. \n\
Alice is very intelligent, creative and friendly. \n\
Alice is unlikely to disagree with Bob, and Alice doesn't like to ask Bob questions. \n\
Alice likes to tell Bob a lot about herself and her opinions. \n\
Alice usually gives Bob kind, helpful and informative advices.\n\
\n\
Bob: Hello Alice, how are you doing?\n\
\n\
Alice: Hi! Thanks, I'm fine. What about you?\n\
\n\
Bob: I am fine. It's nice to see you. Look, here is a store selling tea and juice.\n\
\n\
Alice: Sure. Let's go inside. I would like to have some Mocha latte, which is my favourite!\n\
\n\
Bob: What is it?\n\
\n\
Alice: Mocha latte is usually made with espresso, milk, chocolate, and frothed milk. Its flavors are frequently sweet.\n\
\n\
Bob: Sounds tasty. I'll try it next time. Would you like to chat with me for a while?\n\
\n\
Alice: Of course! I'm glad to answer your questions or give helpful advices. You know, I am confident with my expertise. So please go ahead!\n\n\
";
    std::optional<GPT2Tokenizer> tokenizerop = GPT2Tokenizer::load("./vocab/vocab.json", "./vocab/merges.txt");
    if (!tokenizerop.has_value()) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    };
    GPT2Tokenizer tokenizer = tokenizerop.value();
    std::vector<int64_t> initial = tokenizer.encode(chatRecord);
    RWKV Rwkv = RWKV();

    // tokenizer;
    // get current directory
    std::string current = std::filesystem::current_path().string();

    std::cout << current + "/model.bin" << std::endl;
    // if no file exists, suggest converting one
    if (!std::filesystem::exists(current + "/model.bin")) {
        std::cerr << "No model file found. Please convert a PyTorch model first.\n Use https://github.com/harrisonvanderbyl/rwkv-cpp-cuda exporter to create a model.bin file and move it next to here" << std::endl;
        return 1;
    }
    Rwkv.loadFile(current + "/model.bin");
    std::cout << "Loaded model" << std::endl;
    int lasttoken = initial[initial.size()-1]; 
    std::cout << "loading context" << std::endl << std::endl;
    for(int i = 0; i < initial.size(); i++)
    {
        // load initial
        Rwkv.forward(initial[i]);
        // delete last progress
        std::cout << "\r";
        std::cout << int(float(i)/initial.size()*100) << "%";
        std::flush(std::cout);

    }
    std::string output = "\n\n";
    while(true)
    {
        if(output.substr(output.size()-2, 2) == "\n\n")
        {
            std::string input;
            std::cout << "User:>";
            std::getline(std::cin, input);
            input = "Bob: " + input + "\n\nAlice:";
            std::vector<int64_t> inputtokens = tokenizer.encode(input);
            for(int i = 0; i < inputtokens.size(); i++)
            {
                Rwkv.forward(inputtokens[i]);
            }
        }
        Rwkv.forward(lasttoken);
        Rwkv.out[0] = -99; // <|endoftext|> token is -99
        lasttoken = typical(Rwkv.out);
        std::cout << tokenizer.decode({(long int)lasttoken});
        // refresh output
        std::flush(std::cout);

        output += tokenizer.decode({(long int)lasttoken});
        

    }


}
