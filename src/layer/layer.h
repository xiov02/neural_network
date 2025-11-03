#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuronBase.h"

class Layer {
public:
    int id;
    std::vector<NeuronBase*> neuronLayer;

    virtual const std::vector<float> forward(const std::vector<float>& inputs) = 0;

    std::vector<float> getOutputs();
};

#endif // LAYER_H

