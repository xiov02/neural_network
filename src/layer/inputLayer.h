#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include <vector>
#include <functional>
#include "../neuron/inputNeuron.h"

class InputLayer {
public:
    int id;
    std::vector<InputNeuron*> neuronLayer;

    InputLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction);

    const std::vector<float> forward(const std::vector<float>& inputs);
};

#endif // INPUTLAYER_H

