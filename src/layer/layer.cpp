#include "layer.h"

int Layer::getOutputs(std::vector<double>& outputs) {
    for(size_t i = 0; i < neuronLayer.size(); ++i) {
        outputs[i] = neuronLayer[i]->output;
    }
    return 0;
}