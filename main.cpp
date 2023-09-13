#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>

#include "graph_kernel.h"

#define NUM_ASYNCHRONOUS_ITERATIONS 10  // Number of async loop iterations before attempting to read results back

void checkError(cl_int errorNum, cl_int expected, const char* msg)
{
    if(errorNum != expected)
        std::cerr << "Error occured: " << errorNum << " : " << msg << std::endl;
}


void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex)
{
    graph->vertexCount = numVertices;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->edgeCount = numVertices * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->edgeArray2 = (int*)malloc(graph->edgeCount * sizeof(int));

    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->vertexArray[i] = i * neighborsPerVertex;
    }

    for(int i = 0; i < graph->edgeCount; i++)
    {
        graph->edgeArray[i] = (rand() % graph->vertexCount);
    }

    for(int i = 0; i < graph->vertexCount; ++i) {
        int end;

        if(i == (graph->vertexCount - 1)){
            end = graph->edgeCount;
        }
        else {
            end = graph->vertexArray[i + 1];
        }

        for(int j = graph->vertexArray[i]; j < end; ++j){
            graph->edgeArray2[j] = i;
        }
    }
}

void generateGraph1(GraphData* graph)
{
    graph->vertexCount = 7;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->edgeCount = 16;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->edgeArray2 = (int*)malloc(graph->edgeCount * sizeof(int));

    graph->edgeArray[0] = 1; // 0 begin
    graph->edgeArray[1] = 2; // 0 end
    graph->edgeArray[2] = 0; // 1 begin
    graph->edgeArray[3] = 2;
    graph->edgeArray[4] = 3; // 1 end
    graph->edgeArray[5] = 0; // 2 begin
    graph->edgeArray[6] = 1; //
    graph->edgeArray[7] = 6; // 2 end
    graph->edgeArray[8] = 1; // 3 begin
    graph->edgeArray[9] = 4;  //
    graph->edgeArray[10] = 6;  // 3 end
    graph->edgeArray[11] = 3; // 4 begin
    graph->edgeArray[12] = 5; // 4 end
    graph->edgeArray[13] = 4; // 5 begin/end
    graph->edgeArray[14] = 2; // 6 begin
    graph->edgeArray[15] = 3; // 6 end

    graph->vertexArray[0] = 0;
    graph->vertexArray[1] = 2;
    graph->vertexArray[2] = 5;
    graph->vertexArray[3] = 8;
    graph->vertexArray[4] = 11;
    graph->vertexArray[5] = 13;
    graph->vertexArray[6] = 14;

    for(int i = 0; i < graph->vertexCount; ++i) {
        int end;

        if(i == (graph->vertexCount - 1)){
            end = graph->edgeCount;
        }
        else {
            end = graph->vertexArray[i + 1];
        }

        for(int j = graph->vertexArray[i]; j < end; ++j){
            graph->edgeArray2[j] = i;
        }
    }
}


pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

cl_program loadAndBuildProgram( cl_context gpuContext, const char *fileName )
{
    pthread_mutex_lock(&mutex);

    cl_int errNum;
    cl_program program;

    // Load the OpenCL source code from the .cl file
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *source = srcStdStr.c_str();

    // Create the program for all GPUs in the context
    program = clCreateProgramWithSource(gpuContext, 1, (const char **)&source, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to create OpenCL program");

    // build the program for all devices on the context
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    checkError(errNum, CL_SUCCESS, "Failed to build program");

    pthread_mutex_unlock(&mutex);
    return program;
}


///
/// Gets the id of device with maximal FLOPS from the context (from NVIDIA SDK)
///
cl_device_id getMaxFlopsDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
    int max_flops = 0;

    size_t current_device = 0;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);

    max_flops = compute_units * clock_frequency;
    ++current_device;

    while( current_device < device_count )
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);

        int flops = compute_units * clock_frequency;
        if( flops > max_flops )
        {
            max_flops        = flops;
            max_flops_device = cdDevices[current_device];
        }
        ++current_device;
    }

    free(cdDevices);

    return max_flops_device;
}


///
/// Round the local work size up to the next multiple of the size
///
int roundWorkSizeUp(int groupSize, int globalSize)
{
    int remainder = globalSize % groupSize;
    if (remainder == 0)
    {
        return globalSize;
    }
    else
    {
        return globalSize + groupSize - remainder;
    }
}

void allocateOCLBuffers(cl_context gpuContext, cl_command_queue commandQueue, GraphData *graph,
                        cl_mem *vertexArrayDevice, cl_mem *edgeArrayDevice, cl_mem *edgeArrayDevice2,
                        cl_mem *maskArrayDevice, cl_mem *costArrayDevice, cl_mem *updatingCostArrayDevice,
                        cl_mem *pathsArrayDevice, cl_mem* parentArrayDevice,
                        cl_mem* dependencyArrayDevice, cl_mem* centralityArrayDevice,
                        size_t globalWorkSize)
{
    cl_int errNum;
    cl_mem hostVertexArrayBuffer;
    cl_mem hostEdgeArrayBuffer;
    cl_mem hostEdgeArrayBuffer2;

    // First, need to create OpenCL Host buffers that can be copied to device buffers
    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                           sizeof(int) * graph->vertexCount, graph->vertexArray, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                         sizeof(int) * graph->edgeCount, graph->edgeArray, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    hostEdgeArrayBuffer2 = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                         sizeof(int) * graph->edgeCount, graph->edgeArray2, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    // Now create all of the GPU buffers
    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *edgeArrayDevice2 = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *costArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *updatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *pathsArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");
    *parentArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    *dependencyArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    *centralityArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to allocate buffer");

    // Now queue up the data to be copied to the device
    errNum = clEnqueueCopyBuffer(commandQueue, hostVertexArrayBuffer, *vertexArrayDevice, 0, 0,
                                 sizeof(int) * graph->vertexCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS, "Failed to enqueue buffer");

    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer, *edgeArrayDevice, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS, "Failed to enqueue buffer");

    errNum = clEnqueueCopyBuffer(commandQueue, hostEdgeArrayBuffer2, *edgeArrayDevice2, 0, 0,
                                 sizeof(int) * graph->edgeCount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS, "Failed to enqueue buffer");

    clReleaseMemObject(hostVertexArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer);
    clReleaseMemObject(hostEdgeArrayBuffer2);
}


///
/// Check whether the mask array is empty.  This tells the algorithm whether
/// it needs to continue running or not.
///
bool maskArrayEmpty(int *maskArray, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (maskArray[i] == 1)
        {
            return false;
        }
    }

    return true;
}

///
/// Initialize OpenCL buffers for single run of Dijkstra
///
void initializeOCLBuffers(cl_command_queue commandQueue, cl_kernel initializeKernel, GraphData *graph,
                          size_t maxWorkGroupSize)
{
    cl_int errNum;
    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize = maxWorkGroupSize;
    size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

    errNum = clEnqueueNDRangeKernel(commandQueue, initializeKernel, 1, NULL, &globalWorkSize, &localWorkSize,
                                    0, NULL, NULL);
    checkError(errNum, CL_SUCCESS, "Failed to enqueue buffer (ND range)");
}


void runDijkstra( cl_context context, cl_device_id deviceId, GraphData* graph,
                  int *sourceVertices, int *outResultCosts, int numResults, float* outCentrality)
{
    // Create command queue
    cl_int errNum;
    cl_command_queue commandQueue;
    commandQueue = clCreateCommandQueue( context, deviceId, 0, &errNum );
    checkError(errNum, CL_SUCCESS, "Failed create commnd queue");

    // Program handle
    cl_program program = loadAndBuildProgram( context, "dijkstra.cl" );
    cl_program eb_program = loadAndBuildProgram( context, "edge_betweenness.cl" );

    if((program == nullptr) || (eb_program == nullptr)) {
        return;
    }

    // Get the max workgroup size
    size_t maxWorkGroupSize;
    clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    // checkError(errNum, CL_SUCCESS);
    std::cout << "MAX_WORKGROUP_SIZE: " << maxWorkGroupSize << std::endl;
    std::cout << "Computing '" << numResults << "' results." << std::endl;

    // Set # of work items in work group and total in 1 dimensional range
    size_t localWorkSize = maxWorkGroupSize;
    size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

    cl_mem vertexArrayDevice;
    cl_mem edgeArrayDevice;
    cl_mem edgeArrayDevice2;
    cl_mem maskArrayDevice;
    cl_mem costArrayDevice;
    cl_mem updatingCostArrayDevice;
    cl_mem pathsArrayDevice;
    cl_mem parentArrayDevice;
    cl_mem dependencyArrayDevice;
    cl_mem centralityArrayDevice;

    // Allocate buffers in Device memory
    allocateOCLBuffers( context, commandQueue, graph, &vertexArrayDevice, &edgeArrayDevice, &edgeArrayDevice2,
                        &maskArrayDevice, &costArrayDevice, &updatingCostArrayDevice, &pathsArrayDevice, &parentArrayDevice,
                        &dependencyArrayDevice, &centralityArrayDevice,
                        globalWorkSize);


    // Create the Kernels
    cl_kernel initializeBuffersKernel;
    initializeBuffersKernel = clCreateKernel(program, "initializeBuffers", &errNum);

    // checkError(errNum, CL_SUCCESS);

    // Set the args values and check for errors
    errNum |= clSetKernelArg(initializeBuffersKernel, 0, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 1, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 2, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 3, sizeof(cl_mem), &pathsArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 4, sizeof(cl_mem), &parentArrayDevice);
    errNum |= clSetKernelArg(initializeBuffersKernel, 5, sizeof(cl_mem), &dependencyArrayDevice);

    // 6 set below in loop
    errNum |= clSetKernelArg(initializeBuffersKernel, 7, sizeof(int), &graph->vertexCount);
    checkError(errNum, CL_SUCCESS, "Failed to set kernel arg for init");

    // Kernel 1
    cl_kernel ssspKernel1;
    ssspKernel1 = clCreateKernel(program, "OCL_SSSP_KERNEL1", &errNum);
    checkError(errNum, CL_SUCCESS, "Failed to create kernel 1");
    errNum |= clSetKernelArg(ssspKernel1, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 2, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 3, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 4, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 5, sizeof(cl_mem), &pathsArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 6, sizeof(cl_mem), &parentArrayDevice);
    errNum |= clSetKernelArg(ssspKernel1, 7, sizeof(int), &graph->vertexCount);
    errNum |= clSetKernelArg(ssspKernel1, 8, sizeof(int), &graph->edgeCount);
    checkError(errNum, CL_SUCCESS, "Failed to set kernel 1 arguments");

    // Kernel 2
    cl_kernel ssspKernel2;
    ssspKernel2 = clCreateKernel(program, "OCL_SSSP_KERNEL2", &errNum);
    checkError(errNum, CL_SUCCESS, "Faild to create kernel OCL_SSSP_KERNEL2");
    errNum |= clSetKernelArg(ssspKernel2, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 2, sizeof(cl_mem), &maskArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 3, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 4, sizeof(cl_mem), &updatingCostArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 5, sizeof(cl_mem), &pathsArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 6, sizeof(cl_mem), &parentArrayDevice);
    errNum |= clSetKernelArg(ssspKernel2, 7, sizeof(int), &graph->vertexCount);
    checkError(errNum, CL_SUCCESS, "Failed to set kernel 2 arguments");

    cl_kernel initCentralityBuffers;
    initCentralityBuffers = clCreateKernel(eb_program, "initializeBuffers", &errNum);
    errNum |= clSetKernelArg(initCentralityBuffers, 0, sizeof(cl_mem), &centralityArrayDevice);
    errNum |= clSetKernelArg(initCentralityBuffers, 1, sizeof(int), &graph->vertexCount);
    checkError(errNum, CL_SUCCESS, "Failed to set initCentralityBuffers arguments");

    cl_kernel depKernel;

    int current_depth;

    depKernel = clCreateKernel(eb_program, "calculatePartialDependency", &errNum);
    errNum |= clSetKernelArg(depKernel, 0, sizeof(cl_mem), &vertexArrayDevice);
    errNum |= clSetKernelArg(depKernel, 1, sizeof(cl_mem), &edgeArrayDevice);
    errNum |= clSetKernelArg(depKernel, 2, sizeof(cl_mem), &edgeArrayDevice2);
    errNum |= clSetKernelArg(depKernel, 3, sizeof(cl_mem), &pathsArrayDevice);
    errNum |= clSetKernelArg(depKernel, 4, sizeof(cl_mem), &costArrayDevice);
    errNum |= clSetKernelArg(depKernel, 5, sizeof(cl_mem), &dependencyArrayDevice);
    errNum |= clSetKernelArg(depKernel, 6, sizeof(int), &current_depth);
    // 7 is set below.
    errNum |= clSetKernelArg(depKernel, 8, sizeof(int), &graph->edgeCount);
    checkError(errNum, CL_SUCCESS, "Failed to set calculatePartialDependency arguments");


    cl_kernel finalizeDepKernel;
    finalizeDepKernel = clCreateKernel(eb_program, "finalizeDependency", &errNum);
    errNum |= clSetKernelArg(finalizeDepKernel, 0, sizeof(cl_mem), &centralityArrayDevice);
    errNum |= clSetKernelArg(finalizeDepKernel, 1, sizeof(cl_mem), &dependencyArrayDevice);
    errNum |= clSetKernelArg(finalizeDepKernel, 2, sizeof(int), &graph->vertexCount);
    checkError(errNum, CL_SUCCESS, "Failed to set finalizeDepKernel arguments");
    // 3 is set below.


    int *maskArrayHost = (int*) malloc(sizeof(int) * graph->vertexCount);

    std::vector<int> ordered_vertices;
    ordered_vertices.resize(graph->vertexCount);

    size_t edgeGlobalGroupWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

    for ( int i = 0 ; i < numResults; i++ )
    {
        std::cout << "Source vector = " << sourceVertices[i] << std::endl;

        errNum |= clSetKernelArg(initializeBuffersKernel, 6, sizeof(int), &sourceVertices[i]);
        checkError(errNum, CL_SUCCESS, "Error setting source vertex argument");
        errNum |= clSetKernelArg(depKernel, 7, sizeof(int), &sourceVertices[i]);
        checkError(errNum, CL_SUCCESS, "Error setting source vertex argument2");
        errNum |= clSetKernelArg(finalizeDepKernel, 3, sizeof(int), &sourceVertices[i]);
        checkError(errNum, CL_SUCCESS, "Error setting source vertex argument3");

        // Initialize mask array to false, C and U to infiniti
        initializeOCLBuffers( commandQueue, initializeBuffersKernel, graph, maxWorkGroupSize );

        errNum = clEnqueueNDRangeKernel(commandQueue, initCentralityBuffers, 1, NULL, &globalWorkSize, &localWorkSize,
                                        0, NULL, NULL);
        checkError(errNum, CL_SUCCESS, "Failed to enqueue buffer (ND range)");

        // Read mask array from device -> host
        cl_event readDone;
        errNum = clEnqueueReadBuffer( commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount,
                                      maskArrayHost, 0, NULL, &readDone);
        checkError(errNum, CL_SUCCESS, "EnqueueRead failed");
        clWaitForEvents(1, &readDone);

        current_depth = 0;

        while(!maskArrayEmpty(maskArrayHost, graph->vertexCount))
        {
            // In order to improve performance, we run some number of iterations
            // without reading the results.  This might result in running more iterations
            // than necessary at times, but it will in most cases be faster because
            // we are doing less stalling of the GPU waiting for results.
            for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
            {
                size_t localWorkSize = maxWorkGroupSize;
                size_t globalWorkSize = roundWorkSizeUp(localWorkSize, graph->vertexCount);

                // execute the kernel
                errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel1, 1, 0, &globalWorkSize, &localWorkSize,
                                                0, NULL, NULL);
                checkError(errNum, CL_SUCCESS, "NDRangeKernel failed");

                errNum = clEnqueueNDRangeKernel(commandQueue, ssspKernel2, 1, 0, &globalWorkSize, &localWorkSize,
                                                0, NULL, NULL);
                checkError(errNum, CL_SUCCESS, "NDRangeKernel2 failed");
            }

            current_depth += NUM_ASYNCHRONOUS_ITERATIONS;

            errNum = clEnqueueReadBuffer(commandQueue, maskArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount,
                                         maskArrayHost, 0, NULL, &readDone);
            checkError(errNum, CL_SUCCESS, "clEnqueueReadBuffer2 failed");
            clWaitForEvents(1, &readDone);
        }


        int* results = &outResultCosts[i * graph->vertexCount];

        // Copy the result back
        errNum = clEnqueueReadBuffer(commandQueue, updatingCostArrayDevice, CL_FALSE, 0, sizeof(int) * graph->vertexCount,
                                     results, 0, NULL, &readDone);
        checkError(errNum, CL_SUCCESS, "Copy results failed");
        clWaitForEvents(1, &readDone);

        while(current_depth >= 0) {
            errNum |= clSetKernelArg(depKernel, 6, sizeof(int), &current_depth);
            errNum = clEnqueueNDRangeKernel(commandQueue, depKernel, 1, 0, &globalWorkSize, &localWorkSize,
                                            0, NULL, NULL);
            checkError(errNum, CL_SUCCESS, "NDRangeKernel3 failed");
            --current_depth;
        }

        errNum = clEnqueueNDRangeKernel(commandQueue, finalizeDepKernel, 1, 0, &globalWorkSize, &localWorkSize,
                                        0, NULL, NULL);
        checkError(errNum, CL_SUCCESS, "NDRangeKernel4 failed");
    }


    // Copy the centrality result back
    cl_event readDone;
    errNum = clEnqueueReadBuffer(commandQueue, centralityArrayDevice, CL_FALSE, 0, sizeof(float) * graph->vertexCount,
                                 outCentrality, 0, NULL, &readDone);
    checkError(errNum, CL_SUCCESS, "Copy results failed");
    clWaitForEvents(1, &readDone);
    for(int k = 0; k < graph->vertexCount; ++k) {
        std::cout << "centrality for vector " << k << " : " << outCentrality[k] << std::endl;
    }

    free (maskArrayHost);

    clReleaseMemObject(vertexArrayDevice);
    clReleaseMemObject(edgeArrayDevice);
    clReleaseMemObject(edgeArrayDevice2);
    clReleaseMemObject(maskArrayDevice);
    clReleaseMemObject(costArrayDevice);
    clReleaseMemObject(updatingCostArrayDevice);
    clReleaseMemObject(pathsArrayDevice);
    clReleaseMemObject(parentArrayDevice);
    clReleaseMemObject(dependencyArrayDevice);
    clReleaseMemObject(centralityArrayDevice);

    clReleaseKernel(initializeBuffersKernel);
    clReleaseKernel(ssspKernel1);
    clReleaseKernel(ssspKernel2);
    clReleaseKernel(initCentralityBuffers);
    clReleaseKernel(depKernel);
    clReleaseKernel(finalizeDepKernel);

    clReleaseCommandQueue(commandQueue);
    clReleaseProgram(program);
    clReleaseProgram(eb_program);
    printf("Computed %d results.\n", numResults);
}

void runDijkstraOpenCL( GraphData* graph, int *sourceVertices,
                        int *outResultCosts, int numResults, float* outCentrality) {
    // See what kind of devices are available
    cl_int errNum;
    cl_context gpuContext;

    // create the OpenCL context on available GPU devices
    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

    if(gpuContext != nullptr) {
        std::cout << "Dijkstra OpenCL: Running single GPU version." << std::endl;
        runDijkstra(gpuContext, getMaxFlopsDev(gpuContext), graph, sourceVertices,
                    outResultCosts, numResults, outCentrality);
    }
}




int main(int argc, char** argv) {
    cl_platform_id platform;
    cl_int errNum;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(1, &platform, &numPlatforms);
    printf("Number of OpenCL Platforms: %d\n", numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Failed to find any OpenCL platforms.\n");
        return 1;
    }

    GraphData graph;
    int generateVerts = 1000;
    int generateEdgesPerVert = 2;
    // generateRandomGraph(&graph, generateVerts, generateEdgesPerVert);
    generateGraph1(&graph);

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    // Number of source vertices to search from
    int numSources = 100;

    std::vector<int> sourceVertices{0, 1, 2, 3, 4, 5, 6};
//    sourceVertices.reserve(numSources);
//
//    for(int source = 0; source < numSources; source++)
//    {
//        sourceVertices.push_back(source % graph.vertexCount);
//    }

    int *sourceVertArray = (int*) malloc(sizeof(int) * sourceVertices.size());
    std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

    int* results = (int*) malloc(sizeof(int) * sourceVertices.size() * graph.vertexCount);
    float* outCentrality = (float*) malloc(sizeof(float) * graph.vertexCount);
    runDijkstraOpenCL(&graph, sourceVertArray, results, sourceVertices.size(), outCentrality);

    free(sourceVertArray);
    free(results);
    free(outCentrality);

    return 0;
}
