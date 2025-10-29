#include "multilayerPerceptron.h"

MultilayerPerceptron::MultilayerPerceptron(
    std::vector<int> layerSizes,
    std::function<float(float)> activationFunction
)
    : Network(layerSizes, activationFunction)
{
    id = global_id_counter++;
    id2 = 0;
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