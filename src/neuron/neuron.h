#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    static int global_id_counter;

    int id;
    std::vector<float> weights;
    float bias;
    float output;
    float score;

    Neuron(int numberOfWeight);
};

#endif // NEURON_H
