#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuronBase.h"

class Layer {
public:
    int id;
    std::vector<NeuronBase*> neuronLayer;

    virtual int forward(const std::vector<double>& inputs, std::vector<double>& output) = 0;

    int getOutputs(std::vector<double>& outputs);
};

#endif // LAYER_H

