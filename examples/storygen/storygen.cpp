#include "rwkv.h"
#include <filesystem>

int main(){
    std::string initPrompt = "### Instruction: Write a story/book using the themes and details provided\n\
\n\
### Input:";
    
    

    RWKV Rwkv = RWKV();
    Rwkv.loadTokenizer("./vocab");

    // tokenizer;
    // get current directory
    std::string current = std::filesystem::current_path().string();

    std::cout << current + "/model.bin" << std::endl;
    // if no file exists, suggest converting one
    if (!std::filesystem::exists(current + "/model.bin")) {
        std::cerr << "No model file found. Please convert a PyTorch model first.\n Use https://github.com/harrisonvanderbyl/rwkv-cpp-cuda exporter to create a model.bin file and move it next to here" << std::endl;
        return 1;
    }

    Rwkv.loadFile(current + "/model.bin",2);
    
    std::cout << "Loaded model" << std::endl;
    std::cout << "loading context" << std::endl << std::endl;

    auto lasttoken = Rwkv.loadContext(initPrompt);
    
    RWKVState currentState = Rwkv.state->getSubState();

    std::string output = "\n\n";
    int originalLastToken = lasttoken;
    bool exit = true;
    while(true)
    {
        if(output.length() > 1000 || exit)
        {
            
            if(!exit){
            std::cout << "continue? (y/n):";
            std::string inputs;
            std::getline(std::cin, inputs);
            output = "\n\n";

                if(inputs == "y")
                {
                    continue;
                }

            }
            exit = false;
         

            Rwkv.state->setSubState(currentState);
            lasttoken = originalLastToken;
            std::string input;
            std::cout << "\n\nPress q to return\n\nDescribe the story you want written:>";
            std::getline(std::cin, input);
            input = input + "\n\n### Response:";
            Rwkv.loadContext(input);
        }
        
        Rwkv.forward(lasttoken);
        Rwkv.out[0] = -99; // <|endoftext|> token is -99
        lasttoken = typical(Rwkv.out, 0.8,0.7);
        std::cout << Rwkv.tokenizer->decode({(long int)lasttoken});
        // refresh output
    
        output += Rwkv.tokenizer->decode({(long int)lasttoken});
        std::flush(std::cout);
    }


}
