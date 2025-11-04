#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include <vector>
#include <functional>
#include "../neuron/inputNeuron.h"
#include "layer.h"

class InputLayer : public Layer {
public:
    InputLayer(int numberOfNeurons);

    int forward(const std::vector<float>& inputs, std::vector<float>& output) override;
};

#endif // INPUTLAYER_H

