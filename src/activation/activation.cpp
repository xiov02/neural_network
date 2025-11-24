#include "activation.h"

#include <cmath>
#include <string>

ActivationFunction::ActivationFunction(const std::string activativeFunctionName) {
    if (activativeFunctionName == "sigmoid") {
        function = [](double x) { return 1.0f / (1.0f + std::exp(-x)); };
        derivative = [](double x) {
            double fx = 1.0f / (1.0f + std::exp(-x));
            return fx * (1.0f - fx);
        };
    }
    else if (activativeFunctionName == "relu") {
        function = [](double x) { return x > 0 ? x : 0; };
        derivative = [](double x) { return x > 0 ? 1.0 : 0.0; };
    }
    else {
        // Default to identity function
        function = [](double x) { return x; };
        derivative = [](double x) { return 1.0; };
    }

}