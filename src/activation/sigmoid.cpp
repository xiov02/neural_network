#include "sigmoid.h"

#include <cmath>

Sigmoid::Sigmoid() {}

double Sigmoid::function(double x) {
    return 1.0f / (1.0f + std::exp(-x));
}

double Sigmoid::derivative(double x) {
    double fx = function(x);
    return fx * (1.0f - fx);
}