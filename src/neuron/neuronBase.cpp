#include "neuronBase.h" 

int NeuronBase::global_id_counter = 0; // Define and initialize static member

NeuronBase::NeuronBase()
    : output(0.0f)
{
    id = global_id_counter++;
}