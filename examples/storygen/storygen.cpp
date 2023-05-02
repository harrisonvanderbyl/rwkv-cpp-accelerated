#include "rwkv.h"
#include <filesystem>

int main(){
    std::string initPrompt = "### Instruction: Write a story/book using the themes and details provided\n\
\n\
### Input:";
    std::optional<GPT2Tokenizer> tokenizerop = GPT2Tokenizer::load("./vocab/vocab.json", "./vocab/merges.txt");
    if (!tokenizerop.has_value()) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    };
    GPT2Tokenizer tokenizer = tokenizerop.value();
    std::vector<long long> initial = tokenizer.encode(initPrompt);
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
    // create double* of length [Rwkv.num_layers*Rwkv.num_embed]
    double* xy = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* aa = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* bb = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* pp = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* dd = new double[Rwkv.num_layers*Rwkv.num_embed];
    for(int i = 0; i < Rwkv.num_layers*Rwkv.num_embed; i++)
    {
        xy[i] = Rwkv.statexy[i];
        aa[i] = Rwkv.stateaa[i];
        bb[i] = Rwkv.statebb[i];
        pp[i] = Rwkv.statepp[i];
        dd[i] = Rwkv.statedd[i];
    }


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
            
            if(inputs == "y")
            {
                output = "\n\n";
                continue;
            }
            }
            exit = false;
         

            for (int i = 0; i < Rwkv.num_layers*Rwkv.num_embed; i++)
            {
                Rwkv.statexy[i] = xy[i];
                Rwkv.stateaa[i] = aa[i];
                Rwkv.statebb[i] = bb[i];
                Rwkv.statepp[i] = pp[i];
                Rwkv.statedd[i] = dd[i];
            }
            lasttoken = originalLastToken;
            std::string input;
            std::cout << "\n\nPress q to return\n\nDescribe the story you want written:>";
            std::getline(std::cin, input);
            input = input + "\n\n### Response:";
            std::vector<long long> inputtokens = tokenizer.encode(input);
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
    
        

        output += tokenizer.decode({(long int)lasttoken});
        std::flush(std::cout);
    }


}
