#include<iostream>

#include "hiddenLayer.h"
#include "../activation/activation.h"

int HiddenLayer::global_id_counter = 1; // Define and initialize static member

HiddenLayer::HiddenLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, ActivationFunction activationFunction)
{
    id = global_id_counter++;
    outputs.resize(numberOfNeurons);

    neuronLayer.resize(numberOfNeurons);
    for (int i = 0; i < numberOfNeurons; ++i) {
        neuronLayer[i] = new Neuron(numberOfWeightsPerNeuron, activationFunction);
    }
}

int HiddenLayer::forward(const std::vector<double>& inputs, std::vector<double>& output) {

    for (size_t i = 0; i < neuronLayer.size(); ++i) {
        outputs[i] = neuronLayer[i]->forward(inputs);
        output[i] = outputs[i];
    }
    return 0;
}

int HiddenLayer::getNeuronSize() {
    return neuronLayer.size();
}