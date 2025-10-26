#ifndef INPUTNEURON_H
#define INPUTNEURON_H

class InputNeuron {
public:
    static int global_id_counter;

    int id;
    float output;

    InputNeuron();

    const float forward(float input);
};

#endif // NEURON_H