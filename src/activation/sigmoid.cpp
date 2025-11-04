#include "sigmoid.h"

#include <cmath>

Sigmoid::Sigmoid() {}

float Sigmoid::function(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float Sigmoid::derivative(float x) {
    float fx = function(x);
    return fx * (1.0f - fx);
}