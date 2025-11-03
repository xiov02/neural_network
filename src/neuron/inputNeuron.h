#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include "neuronBase.h"

class InputNeuron : public NeuronBase {
public:
    InputNeuron();

    const float forward(float input);
};

#endif // NEURON_H