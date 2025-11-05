#include "inputNeuron.h"

InputNeuron::InputNeuron(int idNeuron)
    : NeuronBase()
{
    idInputNeuron = idNeuron;
}

const double InputNeuron::forward(const std::vector<double>& inputs) {
    output = inputs[idInputNeuron];
    return output;
}