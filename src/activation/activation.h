#ifndef ACTIVATION_H
#define ACTIVATION_H

class activationFunction {
public:
    virtual float function (float x) = 0;
    virtual float derivative (float x) = 0;
};

#endif // ACTIVATION_H