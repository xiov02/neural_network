#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include "neuronBase.h"

class InputNeuron : public NeuronBase {
public:

    int idInputNeuron;
    InputNeuron(int idNeuron);

    const double forward(const std::vector<double>& inputs) override;
};

#endif // NEURON_H