#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include "neuronBase.h"

class InputNeuron : public NeuronBase {
public:

    int idInputNeuron;
    InputNeuron(int idNeuron);

    const float forward(const std::vector<float>& inputs) override;
};

#endif // NEURON_H