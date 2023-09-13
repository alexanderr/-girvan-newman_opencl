
///
/// Kernel to initialize buffers
///
__kernel void initializeBuffers( __global int *maskArray, __global int *costArray, __global int *updatingCostArray,
                                 __global int *pathsArray, __global int* parentArray,
                                 __global float* dependencyArray,
                                 int sourceVertex, int vertexCount )
{
    // access thread id
    int tid = get_global_id(0);
    parentArray[tid] = -1;
    dependencyArray[tid] = 0.0;

    if (sourceVertex == tid)
    {
        maskArray[tid] = 1;
        costArray[tid] = 0;
        updatingCostArray[tid] = 0;
        pathsArray[tid] = 1;
    }
    else
    {
        maskArray[tid] = 0;
        costArray[tid] = INT_MAX;
        updatingCostArray[tid] = INT_MAX;
        pathsArray[tid] = 0;
    }

}


__kernel  void OCL_SSSP_KERNEL1(__global int *vertexArray, __global int *edgeArray, __global int *maskArray,
                                __global int *costArray, __global int *updatingCostArray,
                                __global int *pathsArray, __global int* parentArray,
                               int vertexCount, int edgeCount )
{
    // access thread id
    int tid = get_global_id(0);

    if ( maskArray[tid] != 0 )
    {
        maskArray[tid] = 0;

        int edgeStart = vertexArray[tid];
        int edgeEnd;
        if (tid + 1 < (vertexCount))
        {
            edgeEnd = vertexArray[tid + 1];
        }
        else
        {
            edgeEnd = edgeCount;
        }

        for(int edge = edgeStart; edge < edgeEnd; edge++)
        {
            int nid = edgeArray[edge];

            if (updatingCostArray[nid] > (costArray[tid] + 1))
            {
                updatingCostArray[nid] = (costArray[tid] + 1);
            }

            if(updatingCostArray[nid] && (updatingCostArray[nid] == (costArray[tid] + 1))) {
                atomic_add(&pathsArray[nid], pathsArray[tid]);
                parentArray[nid] = tid;
            }
        }
    }
}


__kernel  void OCL_SSSP_KERNEL2(__global int *vertexArray, __global int *edgeArray, __global int *maskArray,
                                __global int *costArray, __global int *updatingCostArray,
                                 __global int *pathsArray, __global int* parentArray,
                                int vertexCount)
{
    // access thread id
    int tid = get_global_id(0);


    if (costArray[tid] > updatingCostArray[tid])
    {
        costArray[tid] = updatingCostArray[tid];
        maskArray[tid] = 1;
    }

    updatingCostArray[tid] = costArray[tid];
}



