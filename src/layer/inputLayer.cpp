#include "inputLayer.h"

InputLayer::InputLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction)
{
    id = 0;

    neuronLayer.resize(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; ++i) {
        neuronLayer[i] = new InputNeuron();
    }
}