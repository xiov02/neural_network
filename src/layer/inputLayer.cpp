#include "inputLayer.h"

InputLayer::InputLayer(int numberOfNeurons)
{
    id = 0;

    neuronLayer.resize(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; ++i) {
        neuronLayer[i] = new InputNeuron(i);
    }
}

int InputLayer::forward(const std::vector<double>& inputs, std::vector<double>& output) {
    for (size_t i = 0; i < neuronLayer.size(); ++i) {
        output[i] = neuronLayer[i]->forward(inputs);
    }
    return 0;
}