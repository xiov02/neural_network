#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuronBase.h"

class Layer {
public:
    int id;
    std::vector<NeuronBase*> neuronLayer;

    virtual int forward(const std::vector<float>& inputs, std::vector<float>& output) = 0;

    int getOutputs(std::vector<float>& outputs);
    std::vector<float> getOutputsTemp();
};

#endif // LAYER_H

