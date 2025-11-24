#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <functional>
#include <string>

class ActivationFunction {
public:

    std::function<double(double)> function;
    std::function<double(double)> derivative;

    ActivationFunction(const std::string activativeFunctionName);
};

#endif // ACTIVATION_H