#include "rwkv.h"
#include "tokenizer/tokenizer.h"
#include "samplers/typical.h"
#include <filesystem>
int main(){
    std::cout << "Hello world1" << std::endl;
    
    std::string chatRecord = "The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.\
\n\n\
User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?\
\n\n\
RWKV: It’s Joe Biden; he was sworn in earlier this year.\
\n\n\
User: What year was the French Revolution?\
\n\n\
RWKV: It started in 1789, but it lasted 10 years until 1799.\
\n\n\
User: Can you guess who I might want to marry?\
\n\n\
RWKV: Only if you tell me more about yourself - what are your interests?\
\n\n\
User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?\
\n\n\
RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.\
\n\n\
User";
    std::optional<GPT2Tokenizer> tokenizerop = GPT2Tokenizer::load("./vocab.json", "./merges.txt");
    if (!tokenizerop.has_value()) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    };
    GPT2Tokenizer tokenizer = tokenizerop.value();
    std::vector<int64_t> initial = tokenizer.encode(chatRecord);
    RWKV Rwkv = RWKV();

    // tokenizer;
    // get current directory
    std::string current = std::filesystem::current_path();

    std::cout << current + "/model.bin" << std::endl;
    // if no file exists, suggest converting one
    if (!std::filesystem::exists(current + "/model.bin")) {
        std::cerr << "No model file found. Please convert a PyTorch model first.\n Use https://github.com/harrisonvanderbyl/rwkv-cpp-cuda exporter to create a model.bin file and move it next to here" << std::endl;
        return 1;
    }
    Rwkv.loadFile(current + "/model.bin");
    std::cout << "Loaded model" << std::endl;
    int lasttoken = initial[initial.size()-1]; 

    for(int i = 0; i < initial.size(); i++)
    {
        Rwkv.forward(initial[i]);
    }

    while(true)
    {
        Rwkv.forward(lasttoken);
        lasttoken = typical(Rwkv.out);
        std::cout << tokenizer.decode({(long int)lasttoken});

    }


}
