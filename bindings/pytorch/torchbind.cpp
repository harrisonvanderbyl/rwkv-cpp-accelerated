#include <torch/extension.h>
#include "ATen/ATen.h"
#include "rwkv.h"

RWKV Rwkv = RWKV();

std::vector<torch::Tensor> attachState(torch::Tensor out){

    assert(Rwkv.statexy != NULL);

    // Create a new vector of 5 tensors that have the data_ptr from Rwkv
    torch::Tensor statexy = torch::zeros({Rwkv.num_layers, Rwkv.num_embed}, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor stateaa = torch::zeros({Rwkv.num_layers, Rwkv.num_embed}, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor statebb = torch::zeros({Rwkv.num_layers, Rwkv.num_embed}, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor statepp = torch::zeros({Rwkv.num_layers, Rwkv.num_embed}, torch::TensorOptions().dtype(torch::kDouble));
    torch::Tensor statedd = torch::zeros({Rwkv.num_layers, Rwkv.num_embed}, torch::TensorOptions().dtype(torch::kDouble));

    Rwkv.statexy = (double*)statexy.data_ptr();
    Rwkv.stateaa = (double*)stateaa.data_ptr();
    Rwkv.statebb = (double*)statebb.data_ptr();
    Rwkv.statepp = (double*)statepp.data_ptr();
    Rwkv.statedd = (double*)statedd.data_ptr();
    Rwkv.out = (float*)out.data_ptr();

    return {statexy, stateaa, statebb, statepp, statedd};
}

std::tuple<unsigned long long,unsigned long long> loadWrapper(const std::string& filename){
    Rwkv.loadFile(filename);

    return std::make_tuple(Rwkv.num_layers, Rwkv.num_embed);
}

void cuda_rwkv_wrapper(unsigned long long token){
    Rwkv.forward(token);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
    m.def("load", &loadWrapper, "load");
    m.def("attachState", &attachState, "attachState");

}

TORCH_LIBRARY(rwkv, m) {
    m.def("rwkvc", cuda_rwkv_wrapper);
    m.def("load", loadWrapper);
    m.def("attachState", attachState);
}
