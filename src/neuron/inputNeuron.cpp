#include "inputNeuron.h"

int InputNeuron::global_id_counter = 0; // Define and initialize static member

InputNeuron::InputNeuron()
    : output(0.0f)
{
    id = global_id_counter++;
}
