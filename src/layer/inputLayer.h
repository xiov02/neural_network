#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include <vector>
#include <functional>
#include "../neuron/inputNeuron.h"
#include "layer.h"

class InputLayer : public Layer {
public:
    std::vector<InputNeuron*> neuronLayer;

    InputLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction);

    const std::vector<float> forward(const std::vector<float>& inputs) override;
};

#endif // INPUTLAYER_H

