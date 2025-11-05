#ifndef ACTIVATION_H
#define ACTIVATION_H

class activationFunction {
public:
    virtual double function (double x) = 0;
    virtual double derivative (double x) = 0;
};

#endif // ACTIVATION_H