#include <stdio.h>
#include "cupy_cutensor.h"

void _cutensor_alloc_handle(cutensorHandle_t **handle)
{
    *handle = (cutensorHandle_t*) malloc( sizeof(cutensorHandle_t) );
}

void _cutensor_free_handle(cutensorHandle_t *handle)
{
    free(handle);
}

void _cutensor_alloc_tensor_descriptor(cutensorTensorDescriptor_t **desc)
{
    *desc = (cutensorTensorDescriptor_t*) malloc( sizeof(cutensorTensorDescriptor_t) );
}

void _cutensor_free_tensor_descriptor(cutensorTensorDescriptor_t *desc)
{
    free(desc);
}

void _cutensor_alloc_contraction_descriptor(cutensorContractionDescriptor_t **desc)
{
    *desc = (cutensorContractionDescriptor_t*) malloc( sizeof(cutensorContractionDescriptor_t) );
}

void _cutensor_free_contraction_descriptor(cutensorContractionDescriptor_t *desc)
{
    free(desc);
}

void _cutensor_alloc_contraction_plan(cutensorContractionPlan_t **plan)
{
    *plan = (cutensorContractionPlan_t*) malloc( sizeof(cutensorContractionPlan_t) );
}

void _cutensor_free_contraction_plan(cutensorContractionPlan_t *plan)
{
    free(plan);
}

void _cutensor_alloc_contraction_find(cutensorContractionFind_t **find)
{
    *find = (cutensorContractionFind_t*) malloc( sizeof(cutensorContractionFind_t) );
}

void _cutensor_free_contraction_find(cutensorContractionFind_t *find)
{
    free(find);
}
