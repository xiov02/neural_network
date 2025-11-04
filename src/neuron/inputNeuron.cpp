#include "inputNeuron.h"

InputNeuron::InputNeuron(int idNeuron)
    : NeuronBase()
{
    idInputNeuron = idNeuron;
}

const float InputNeuron::forward(const std::vector<float>& inputs) {
    output = inputs[idInputNeuron];
    return output;
}