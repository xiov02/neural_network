#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include <vector>
#include <functional>
#include "../neuron/inputNeuron.h"
#include "layer.h"

class InputLayer : public Layer {
public:
    InputLayer(int numberOfNeurons);

    const std::vector<float> forward(const std::vector<float>& inputs) override;
};

#endif // INPUTLAYER_H

