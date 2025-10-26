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
};

#endif // INPUTLAYER_H

