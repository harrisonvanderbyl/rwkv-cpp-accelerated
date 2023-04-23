#include "rwkv.h"

int main(){

    RWKV Rwkv = RWKV();

    Rwkv.loadFile("./export/rwkv.bin");
    int lasttoken = 127;
    while(true)
    {
        Rwkv.forward(lasttoken);
        std::cout << Rwkv.out[0] << "\n";
    }


}
