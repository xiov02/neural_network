#include "multilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(std::vector<int> layerSizes, std::function<float(float)> activationFunction)
    : inputLayer(layerSizes[0], 0, activationFunction),
      outputLayer(layerSizes.back(), layerSizes[layerSizes.size()-2], activationFunction)
{
    HiddenLayer::global_id_counter = 1; // Reset hidden layer ID counter
    // Create hidden layers
    for (size_t i = 1; i < layerSizes.size() - 1; ++i) {
        int numberOfWeightsPerNeuron = layerSizes[i - 1];
        hiddenLayers.emplace_back(layerSizes[i], numberOfWeightsPerNeuron, activationFunction);
    }
}

const std::vector<float> MultilayerPerceptron::forward(const std::vector<float>& inputs) {
    std::vector<float> currentOutputs = inputLayer.forward(inputs);

    // Forward through hidden layers
    for (HiddenLayer layer : hiddenLayers) {
        currentOutputs = layer.forward(currentOutputs);
    }

    // Forward through output layer
    currentOutputs = outputLayer.forward(currentOutputs);

    return currentOutputs;
}