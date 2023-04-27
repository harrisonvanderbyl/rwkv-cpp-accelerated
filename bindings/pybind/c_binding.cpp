#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "rwkv.h"

namespace py = pybind11;

RWKV Rwkv = RWKV();

py::list initState(){

    py::array_t<double> statexy_array = py::array_t<double>(Rwkv.num_layers * Rwkv.num_embed);
    py::buffer_info statexy_buf = statexy_array.request();

    py::array_t<double> stateaa_array = py::array_t<double>(Rwkv.num_layers * Rwkv.num_embed);
    py::buffer_info stateaa_buf = stateaa_array.request();

    py::array_t<double> statebb_array = py::array_t<double>(Rwkv.num_layers * Rwkv.num_embed);
    py::buffer_info statebb_buf = statebb_array.request();

    py::array_t<double> statepp_array = py::array_t<double>(Rwkv.num_layers * Rwkv.num_embed);
    py::buffer_info statepp_buf = statepp_array.request();

    py::array_t<double> statedd_array = py::array_t<double>(Rwkv.num_layers * Rwkv.num_embed);
    py::buffer_info statedd_buf = statedd_array.request();

    Rwkv.statexy  = (double*)statexy_buf.ptr;
    Rwkv.stateaa  = (double*)stateaa_buf.ptr;
    Rwkv.statebb  = (double*)statebb_buf.ptr;
    Rwkv.statepp  = (double*)statepp_buf.ptr;
    Rwkv.statedd  = (double*)statedd_buf.ptr;

    for (unsigned long long i = 0; i < Rwkv.num_layers*Rwkv.num_embed; i++) {
        Rwkv.statexy[i] = 0;
        Rwkv.stateaa[i] = 0;
        Rwkv.statebb[i] = 0;
        Rwkv.statepp[i] = 0;
        Rwkv.statedd[i] = 0;
    }

    py::list return_list;
    return_list.append(statexy_array);
    return_list.append(stateaa_array);
    return_list.append(statebb_array);
    return_list.append(statepp_array);
    return_list.append(statedd_array);
    return return_list;
}

py::list getRwkvState() {
    py::list return_list;

    return_list.append(Rwkv.statexy);
    return_list.append(Rwkv.stateaa);
    return_list.append(Rwkv.statebb);
    return_list.append(Rwkv.statepp);
    return_list.append(Rwkv.statedd);

    return return_list;

}

py::array_t<float> initOutput() {
    py::array_t<float> out_array = py::array_t<float>(50277);
    py::buffer_info out_buf = out_array.request();

    Rwkv.out  = (float*)out_buf.ptr;

    for (unsigned long long i = 0; i < 50277; i++) {
	    Rwkv.out[i] = 0;
	}

    return out_array;
}

float* getRwkvOutput() {
	return Rwkv.out;

}

std::tuple<int64_t, int64_t> loadWrapper(const std::string& filename){
    Rwkv.loadFile(filename);

    return std::make_tuple(Rwkv.num_layers, Rwkv.num_embed);
}

void cuda_rwkv_wrapper(int64_t token){
    Rwkv.forward(token);
}


PYBIND11_MODULE(rwkv, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
    m.def("load", &loadWrapper, "load");
    m.def("initState", &initState, "initState");
    m.def("initOutput", &initOutput, "initOutput");

    m.def("getRwkvOutput", &getRwkvOutput, "getRwkvOutput");
    m.def("getRwkvState", &getRwkvState, "getRwkvState");

}

