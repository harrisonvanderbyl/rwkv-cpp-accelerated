#include <pybind11/pybind11.h>

#include "rwkv.h"

namespace py = pybind11;

RWKV Rwkv = RWKV();

std::vector<double*> initState(){

    assert(Rwkv.statexy != NULL);

    // Create a new vector of 5 tensors that have the data_ptr from Rwkv
    double* statexy = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* stateaa = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* statebb = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* statepp = new double[Rwkv.num_layers*Rwkv.num_embed];
    double* statedd = new double[Rwkv.num_layers*Rwkv.num_embed];

    for (unsigned long long i = 0; i < Rwkv.num_layers*Rwkv.num_embed; i++) {
        statexy[i] = 0;
        stateaa[i] = 0;
        statebb[i] = 0;
        statepp[i] = 0;
        statedd[i] = 0;
    }

    for (unsigned long long i = 0; i < 50277; i++) {
        Rwkv.out[i] = 0;
    }    

    return {statexy, stateaa, statebb, statepp, statedd};
}

std::tuple<int64_t, int64_t> loadWrapper(const std::string& filename){
    Rwkv.loadFile(filename);

    return std::make_tuple(Rwkv.num_layers, Rwkv.num_embed);
}

void cuda_rwkv_wrapper(int64_t token){
    Rwkv.forward(token);
}


PYBIND11_MODULE(c_binding, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
    m.def("load", &loadWrapper, "load");
    m.def("initState", &initState, "initState");

}

