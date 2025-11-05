#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include <vector>
#include <functional>
#include "../neuron/neuron.h"
#include "layer.h"

class HiddenLayer : public Layer {
public:
    static int global_id_counter;

    std::vector<Neuron*> neuronLayer;

    HiddenLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<double(double)> activationFunction);
    
    int forward(const std::vector<double>& inputs, std::vector<double>& output) override;
};

#endif // HIDDENLAYER_H

