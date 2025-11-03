#include "inputNeuron.h"

InputNeuron::InputNeuron()
    : NeuronBase()
{
}

const float InputNeuron::forward(const float input) {
    output = input;
    return output;
}