#include "inputLayer.h"

InputLayer::InputLayer(int numberOfNeurons)
{
    id = 0;

    neuronLayer.resize(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; ++i) {
        neuronLayer[i] = new InputNeuron();
    }
}

const std::vector<float> InputLayer::forward(const std::vector<float>& inputs) {    
    return inputs;
}