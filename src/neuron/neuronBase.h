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

    virtual const float forward(const std::vector<float>& inputs) = 0;
};

#endif // NEURONBASE_H