#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#include "rwkv.h"

namespace py = pybind11;


void* initRwkv() {
    return new RWKV();

}


void* initTokenizer(const std::string& vocab_filename, const std::string& merges_filename) {
    std::optional<GPT2Tokenizer> tokenizerop = GPT2Tokenizer::load(vocab_filename, merges_filename);

    if (!tokenizerop.has_value()) {
	std::cerr << "Failed to load tokenizer" << std::endl;
	throw py::value_error("Failed to load tokenizer");
    };

    GPT2Tokenizer* tokenizer = new GPT2Tokenizer(tokenizerop.value());

    return tokenizer;
}


void initRwkvOutput(void* rwkvp) {
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    py::array_t<float> out_array = py::array_t<float>(50277);
    py::buffer_info out_buf = out_array.request();

    rwkv->out  = (float*)out_buf.ptr;

    for (unsigned long long i = 0; i < 50277; i++) {
	    rwkv->out[i] = 0;
	}

}


void initRwkvState(void* rwkvp){
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    int size = rwkv->num_layers * rwkv->num_embed;

    rwkv->statexy = new double[size];
    rwkv->stateaa = new double[size];
    rwkv->statebb = new double[size];
    rwkv->statepp = new double[size];
    rwkv->statedd = new double[size];

    for (unsigned long long i = 0; i < size; i++) {
	rwkv->statexy[i] = 0;
	rwkv->stateaa[i] = 0;
	rwkv->statebb[i] = 0;
	rwkv->statepp[i] = 0;
	rwkv->statedd[i] = 0;
    }

}


py::array_t<float> getRwkvOutput(void* rwkvp) {
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    py::array_t<float> out_array = py::array_t<float>(50277);
    py::buffer_info out_buf = out_array.request();

    float* out_buf_ptr = (float*)out_buf.ptr;
    for (unsigned long long i = 0; i < 50277; i++) {
        out_buf_ptr[i] = rwkv->out[i];
    }

    return out_array;
}


py::list getRwkvState(void* rwkvp) {
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    py::array_t<double> statexy_array = py::array_t<double>(50277);
    py::buffer_info statexy_buf = statexy_array.request();

    py::array_t<double> stateaa_array = py::array_t<double>(50277);
    py::buffer_info stateaa_buf = stateaa_array.request();

    py::array_t<double> statebb_array = py::array_t<double>(50277);
    py::buffer_info statebb_buf = statebb_array.request();

    py::array_t<double> statepp_array = py::array_t<double>(50277);
    py::buffer_info statepp_buf = statepp_array.request();

    py::array_t<double> statedd_array = py::array_t<double>(50277);
    py::buffer_info statedd_buf = statedd_array.request();



    double* statexy_buf_ptr = (double*)statexy_buf.ptr;
    double* stateaa_buf_ptr = (double*)stateaa_buf.ptr;
    double* statebb_buf_ptr = (double*)statebb_buf.ptr;
    double* statepp_buf_ptr = (double*)statepp_buf.ptr;
    double* statedd_buf_ptr = (double*)statedd_buf.ptr;

    for (unsigned long long i = 0; i < 50277; i++) {
        statexy_buf_ptr[i] = rwkv->statexy[i];
        stateaa_buf_ptr[i] = rwkv->stateaa[i];
        statebb_buf_ptr[i] = rwkv->statebb[i];
        statepp_buf_ptr[i] = rwkv->statepp[i];
        statedd_buf_ptr[i] = rwkv->statedd[i];
    }

    py::list output_list;
    output_list.append(statexy_array);
    output_list.append(stateaa_array);
    output_list.append(statebb_array);
    output_list.append(statepp_array);
    output_list.append(statedd_array);

    return output_list;
}


std::vector<int64_t> tokenizerEncode(void* tokenizerp, std::string tok_str) {
    GPT2Tokenizer* tokenizer = static_cast<GPT2Tokenizer*>(tokenizerp);

    return tokenizer->encode(tok_str);
}


std::string tokenizerDecode(void* tokenizerp, int token) {
    GPT2Tokenizer* tokenizer = static_cast<GPT2Tokenizer*>(tokenizerp);

    return tokenizer->decode({(long int)token});
}

int typicalSample(void* rwkvp) {
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    return typical(rwkv->out);
}


std::tuple<int64_t, int64_t> loadWrapper(void* rwkvp, const std::string& filename){
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);
    
    rwkv->loadFile(filename);
    return std::make_tuple(rwkv->num_layers, rwkv->num_embed);
}


void modelForward(void* rwkvp, int64_t token){
    RWKV* rwkv = static_cast<RWKV*>(rwkvp);

    rwkv->forward(token);
}


PYBIND11_MODULE(rwkv, m) {
    m.def("initRwkv", &initRwkv, "initRwkv");
    m.def("modelForward", &modelForward, "rwkvc");
    m.def("loadModel", &loadWrapper, "load");

    m.def("initState", &initRwkvState, "initState");
    m.def("getState", &getRwkvState, "getRwkvState");

    m.def("initOutput", &initRwkvOutput, "initOutput");
    m.def("getOutput", &getRwkvOutput, "getRwkvOutput");

    m.def("initTokenizer", &initTokenizer, "getRwkvOutput");
    m.def("tokenizerEncode", &tokenizerEncode, "tokenizerEncode");
    m.def("tokenizerDecode", &tokenizerDecode, "tokenizerDecode");

    m.def("typicalSample", &typicalSample, "typicalSample");

}

