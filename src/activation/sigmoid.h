#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation.h"

class Sigmoid : public activationFunction {
public:
    double function(double x) override;
    double derivative(double x) override;

    Sigmoid();
};

#endif // SIGMOID_H
