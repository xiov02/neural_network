#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation.h"

class Sigmoid : public activationFunction {
public:
    float function(float x) override;
    float derivative(float x) override;

    Sigmoid();
};

#endif // SIGMOID_H
