#ifndef DIJKSTRA_KERNEL_H
#define DIJKSTRA_KERNEL_H


#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif



struct GraphData {
    // (V) This contains a pointer to the edge list for each vertex
    int *vertexArray;

    // Vertex count
    int vertexCount;

    // (E) This contains pointers to the vertices that each edge is attached to in CSR matrix format.
    int *edgeArray;

    // This contains pointers to the vertices that each edge is attached to in COO matrix format.
    int *edgeArray2;

    // Edge count
    int edgeCount;
};

struct OCLCommunityDetection {
    GraphData* graph;
    cl_program dijkstraProgram;
    cl_program edgeBetweennessProgram;

    cl_mem vertexArrayDevice;
    cl_mem edgeArrayDevice;
    cl_mem edgeArrayDevice2;
    cl_mem maskArrayDevice;
    cl_mem costArrayDevice;
    cl_mem updatingCostArrayDevice;
    cl_mem sigmaArrayDevice;
    cl_mem predecessorArrayDevice;
    cl_mem dependencyArrayDevice;
    cl_mem centralityArrayDevice;


    ~OCLCommunityDetection() {
        clReleaseMemObject(vertexArrayDevice);
        clReleaseMemObject(edgeArrayDevice);
        clReleaseMemObject(edgeArrayDevice2);
        clReleaseMemObject(maskArrayDevice);
        clReleaseMemObject(costArrayDevice);
        clReleaseMemObject(updatingCostArrayDevice);
        clReleaseMemObject(sigmaArrayDevice);
        clReleaseMemObject(predecessorArrayDevice);
        clReleaseMemObject(dependencyArrayDevice);
        clReleaseMemObject(centralityArrayDevice);

        clReleaseProgram(dijkstraProgram);
        clReleaseProgram(edgeBetweennessProgram);
    }

};




#endif //DIJKSTRA_KERNEL_H
