// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
//#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include <iostream>
#include "../../ecc/burg_ecc.h"

namespace py = pybind11;
using namespace aloha::ecc;

constexpr int MAX_AUDIO_CHANS = 8;

template<int n_audio_channels>
class wrapper_class
{
    public:
    wrapper_class(BurgEccParameters& parameters)
    {
        wrapped_object = new BurgErrorConcealer<n_audio_channels>(parameters);
    }

    ~wrapper_class()
    {
        wrapped_object->~BurgErrorConcealer();
    }

    void set_mode(EccMode mode)
    {
        wrapped_object->set_mode(mode);
    }

    EccMode mode()
    {
        return wrapped_object->mode();
    }

    void prepare_to_play(float sampling_rate, int audio_buffer_size)
    {
        buffer_size = audio_buffer_size;
        wrapped_object->prepare_to_play(sampling_rate, audio_buffer_size);
    }

    void process(py::array_t<float, py::array::c_style | py::array::forcecast> inputs, py::array_t<float, py::array::c_style | py::array::forcecast> outputs, bool is_packet_valid)
    {
        if (inputs.ndim() > 2)
            throw std::runtime_error("Input cannot have more than 2 dimensions");
        if (outputs.ndim() > 2)
            throw std::runtime_error("Input cannot have more than 2 dimensions");

        //std::cout << toascii(n_audio_channels) << std::endl;

        const float * in_ptr = inputs.data(0);
        float * out_ptr = outputs.mutable_data(0);

        // if(true)    {
        //     for (int i=0; i < inputs.size()/inputs.ndim(); i++) {
        //         std::cout << "Size: " << inputs.size() << " Value: " << *inputs.data(i) << " Address: " << inputs.data(i) << std::endl;
        //         //std::cout << "Value: " << *inputs.data(i) << std::endl;
        //         //std::cout << "Address: " << inputs.data(i) << std::endl;
        //     }
        //     temp = false;
        // }

        for (int i=0; i < n_audio_channels; i++)
        {
            //std::cout << "i " << toascii(i) << std::endl;
            vector_of_pointers_in[i] = in_ptr + i*buffer_size;
            vector_of_pointers_out[i] = out_ptr + i*buffer_size;
        }
        //std::cout << "Pre process" << std::endl;
        wrapped_object->process(vector_of_pointers_in, vector_of_pointers_out, is_packet_valid);
    }
    private:
    BurgErrorConcealer<n_audio_channels>* wrapped_object;
    const float* vector_of_pointers_in[MAX_AUDIO_CHANS];
    float* vector_of_pointers_out[MAX_AUDIO_CHANS];
    int buffer_size;
};

PYBIND11_MODULE(ecc_external, m) {
    constexpr int n_audio_channels = 2;
    py::class_<wrapper_class<n_audio_channels>, std::shared_ptr<wrapper_class<n_audio_channels>>> bec(m, "BurgErrorConcealer");

    bec.def(py::init<BurgEccParameters &>())
       .def("set_mode", &wrapper_class<n_audio_channels>::set_mode)
       .def("mode", &wrapper_class<n_audio_channels>::mode)
       .def("prepare_to_play", &wrapper_class<n_audio_channels>::prepare_to_play)
       .def("process", &wrapper_class<n_audio_channels>::process);

    py::class_<BurgEccParameters, std::shared_ptr<BurgEccParameters>>(m, "BurgEccParameters")
        .def(py::init<>())
        .def_readwrite("mid_filter_length", &BurgEccParameters::mid_filter_length)
        .def_readwrite("mid_cross_fade_time", &BurgEccParameters::mid_cross_fade_time)  
        .def_readwrite("side_filter_length", &BurgEccParameters::side_filter_length)
        .def_readwrite("side_cross_fade_time", &BurgEccParameters::side_cross_fade_time);
    
    py::enum_<EccMode>(m, "EccMode")
        .value("STEREO", EccMode::STEREO)
        .value("DUAL MONO", EccMode::DUAL_MONO)
        .value("MONO L", EccMode::MONO_L)
        .value("MONO R", EccMode::MONO_R)
        .value("NONE", EccMode::NONE);
    
    m.doc() = "External C++ ecc module wrapped for python use";
}