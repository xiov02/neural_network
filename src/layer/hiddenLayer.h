#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuron.h"

class HiddenLayer {
public:
    static int global_id_counter;

    int id;
    std::vector<Neuron*> neuronLayer;

    HiddenLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction);
};

#endif // HIDDENLAYER_H

