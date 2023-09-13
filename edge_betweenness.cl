void atomic_add_global(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;

    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void initializeBuffers(__global float *centrality, int vertexCount) {
    // access thread id
    int v = get_global_id(0);

    if(v >= vertexCount)
        return;

    centrality[v] = 0.0;
}

__kernel void finalizeDependency(__global float *centrality, __global float* dependency, int vertexCount, int sourceVertex) {
    // access thread id
    int v = get_global_id(0);

    if(v >= vertexCount)
        return;

    if(v == sourceVertex)
        return;

    // shortest paths are counted twice.
    centrality[v] += dependency[v] / 2;
}


__kernel void calculatePartialDependency(__global int *vertexArray, __global int *edgeArray, __global int* edgeArray2,
                                __global int *pathsArray, __global int *costArray, __global float *dependencyArray,
                                int currentDepth, int sourceVertex, int edgeCount)
{
    // access thread id
    int tid = get_global_id(0);

    if(tid >= edgeCount)
        return;

    int v = edgeArray2[tid];

    if(costArray[v] != currentDepth)
        return;

    int w = edgeArray[tid];

    // Do we pass v to get to w?
    if(costArray[w] == (costArray[v] + 1)) {
        if(pathsArray[w] != 0) {
            float sigma_v = (float)pathsArray[v];
            float sigma_w = (float)pathsArray[w];
             atomic_add_global(dependencyArray + v, (sigma_v * 1.0 / sigma_w) * (1 + dependencyArray[w]));
        }
    }
}