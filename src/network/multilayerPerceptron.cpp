#include "multilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(std::vector<int> layerSizes, std::function<float(float)> activationFunction)
{
    // Initialize input layer
    inputLayer = InputLayer(layerSizes[0], 0, activationFunction);
    // Initialize hidden layers
    hiddenLayers.clear();
    for (size_t i = 1; i < layerSizes.size() - 1; ++i) {
        hiddenLayers.emplace_back(layerSizes[i], layerSizes[i - 1], activationFunction);
    }
    // Initialize output layer
    outputLayer = HiddenLayer(layerSizes.back(), layerSizes[layerSizes.size() - 2], activationFunction);

    id = global_id_counter++;
}

const std::vector<float> MultilayerPerceptron::forward(const std::vector<float>& inputs) {
    std::vector<float> currentOutputs = inputLayer.forward(inputs);

    // Forward through hidden layers
    for (HiddenLayer& layer : hiddenLayers) {
        currentOutputs = layer.forward(currentOutputs);
    }

    // Forward through output layer
    currentOutputs = outputLayer.forward(currentOutputs);

    return currentOutputs;
}