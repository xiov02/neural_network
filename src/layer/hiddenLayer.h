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

    HiddenLayer(int numberOfNeurons, int numberOfWeightsPerNeuron, std::function<float(float)> activationFunction);
    
    const std::vector<float> forward(const std::vector<float>& inputs) override;
};

#endif // HIDDENLAYER_H

