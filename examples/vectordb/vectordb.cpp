#include "rwkv.h"
#include <filesystem>

int main(){
 
    RWKV Rwkv = RWKV();
    Rwkv.loadTokenizer("./vocab");
    
    // if no file exists, suggest converting one
    if (!std::filesystem::exists("./model.bin")) {
        std::cerr << "No model file found. Please convert a PyTorch model first.\n Use https://github.com/harrisonvanderbyl/rwkv-cpp-cuda exporter to create a model.bin file and move it next to here" << std::endl;
        return 1;
    }

    Rwkv.loadFile("./model.bin",5);
    
    std::cout << "Loaded model" << std::endl;

    auto emptyState = Rwkv.emptyState();

    std::vector<std::string> facts = {
        "Alice is a person",
        "Nvidia is a company",
        "The grand canyon is a place",
    };

    std::vector<RWKVState*> states;

    for(int i = 0; i < facts.size(); i++)
    {
        Rwkv.loadContext(facts[i]);
        RWKVState state = Rwkv.state->getSubState();
        states.push_back(new RWKVState(state));
        Rwkv.state->setSubState(emptyState);
    }

    while(1){
        std::string input;
        std::cout << "Question:>";
        std::getline(std::cin, input);
        
        Rwkv.state->setSubState(emptyState);
        Rwkv.loadContext(input);
        auto state = Rwkv.state->getSubState();

        for(int i = 0; i < facts.size(); i++)
        {
            auto a = states[i]->statedd;
            auto b = state.statedd;
            float diff = 0;
            float euclid = 0;
            for(int j = 0; j < Rwkv.num_embed*Rwkv.num_layers; j++)
            {
                diff += std::abs(a[j] - b[j])/Rwkv.num_embed;
                euclid += std::pow(a[j] - b[j],2)/Rwkv.num_embed;
            }
            std::cout << "Diff: '" << facts[i] << "' is " << diff << ":" << sqrt(euclid) << std::endl;
        }
    }

    


}
