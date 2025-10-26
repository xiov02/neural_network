#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuron.h"

class Layer {
public:
    int id;

    virtual const std::vector<float> forward(const std::vector<float>& inputs) = 0;
};

#endif // LAYER_H

