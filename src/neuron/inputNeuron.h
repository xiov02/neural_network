#ifndef INPUTNEURON_H
#define INPUTNEURON_H

class InputNeuron {
public:
    static int global_id_counter;

    int id;
    float output;

    InputNeuron();
};

#endif // NEURON_H