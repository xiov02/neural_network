#ifndef NEURONBASE_H
#define NEURONBASE_H

#include <vector>

class NeuronBase {

protected:
    NeuronBase();
public:
    virtual ~NeuronBase() = default;

    static int global_id_counter;
    int id;

    float output;
};

#endif // NEURONBASE_H