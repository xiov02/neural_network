#include "hiddenLayer.h"

int HiddenLayer::global_id_counter = 1; // Define and initialize static member

HiddenLayer::HiddenLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction)
{
    id = global_id_counter++;

    neuronLayer.resize(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; ++i) {
        neuronLayer[i] = new Neuron(numberOfWeightsPerNeuron, activationFunction);
    }
}

const std::vector<float> HiddenLayer::forward(const std::vector<float>& inputs) {
    std::vector<float> outputs;
    outputs.reserve(neuronLayer.size());

    for (Neuron* neuron : neuronLayer) {
        outputs.push_back(neuron->forward(inputs));
    }

    return outputs;
}