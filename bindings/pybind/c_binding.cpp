#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "rwkv.h"

namespace py = pybind11;


RWKV* initRwkv() {
    return &RWKV();

}

void initRwkvState(RWKV &rwkv){
    return return_list;
}


void initRwkvOutput(RWKV &rwkv) {
    py::array_t<float> out_array = py::array_t<float>(50277);
    py::buffer_info out_buf = out_array.request();

    rwkv.out  = (float*)out_buf.ptr;

    for (unsigned long long i = 0; i < 50277; i++) {
	    rwkv.out[i] = 0;
	}

    return out_array;
}


py::array_t<float> getRwkvOutput(RWKV &rwkv) {
    py::array_t<float> out_array = py::array_t<float>(50277);
    py::buffer_info out_buf = out_array.request();

    for (unsigned long long i = 0; i < 50277; i++) {
        out_buf.ptr[i] = rwkv.out[i];
    }

    return out_array;

}

py::list getRwkvState(RWKV &rwkv) {



}

std::tuple<int64_t, int64_t> loadWrapper(RWKV &rwkv, const std::string& filename){
    rwkv.loadFile(filename);
    return std::make_tuple(Rwkv.num_layers, Rwkv.num_embed);
}

void modelForward(RWKV &rwkv, int64_t token){
    Rwkv.forward(token);
}


PYBIND11_MODULE(rwkv, m) {
    m.def("modelForward", &cuda_rwkv_wrapper, "rwkvc");
    m.def("loadModel", &loadWrapper, "load");

    m.def("initState", &initState, "initState");
    m.def("getRwkvState", &getRwkvState, "getRwkvState");

    m.def("initOutput", &initOutput, "initOutput");
    m.def("getRwkvOutput", &getRwkvOutput, "getRwkvOutput");

    m.def("loadTokenizer", &getRwkvOutput, "getRwkvOutput");
    m.def("tokenizerEncode", &getRwkvOutput, "getRwkvOutput");
    m.def("tokenizerDecode", &getRwkvOutput, "getRwkvOutput");

    m.def("typicalSample", &getRwkvOutput, "getRwkvOutput");

}

