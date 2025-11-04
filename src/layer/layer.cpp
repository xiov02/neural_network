#include "layer.h"

int Layer::getOutputs(std::vector<float>& outputs) {
    for(size_t i = 0; i < neuronLayer.size(); ++i) {
        outputs[i] = neuronLayer[i]->output;
    }
    return 0;
}

std::vector<float> Layer::getOutputsTemp() {
    std::vector<float> outputs;
    for (const auto& neuronPtr : this->neuronLayer) {
        outputs.push_back(neuronPtr->output);
    }

    return outputs;
}