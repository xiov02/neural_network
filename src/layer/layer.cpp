#include "layer.h"

std::vector<float> Layer::getOutputs() {
    std::vector<float> outputs;
    for (const auto& neuronPtr : this->neuronLayer) {
        outputs.push_back(neuronPtr->output);
    }

    return outputs;
}