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

    double output;

    virtual const double forward(const std::vector<double>& inputs) = 0;
};

#endif // NEURONBASE_H