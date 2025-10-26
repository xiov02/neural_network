#include <iostream>

#include "neuron/neuron.h"

int main() {

    int x = 10;
    Neuron neuron2 = Neuron(x);

    printf("Neuron2 ID = %d\n",neuron2.id);

    for (size_t i = 0; i < x; ++i) {
        printf("Neuron2 Weight[%zu] = %f\n", i, neuron2.weights[i]);
    }

    return 0;
}