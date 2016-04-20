/* File: flow.cpp
 *
 * Description: This file contains the program to upload the 3D mesh points of
 *     frames 1 and 2 of a wing surface. It finds the flow of each point in the 
 *     frame 2 mesh by finding the k closest frame 1 mesh points, getting the 
 *     flow from frame 1 to frame 2 of those k closest points, and finding a 
 *     weighted average of those flows, based on their distances to the frame 2 
 *     mesh point. The mesh points and their corresponding flows are then outputted
 *     to file.
 *
 * Author: Ming Guo
 * Created: 9/1/11
 */

#include "wing.h"
#include "flow.h"

// K is the number of nearest points in the successive frame's mesh to calculate 
// the velocity vector for each mesh point in the current frame from
#define K 10

// struct to store the deltas of a 3D Vector
typedef struct {
    double deltaX;
    double deltaY;
    double deltaZ;
} Vector;

/* Function: getDistanceBetweenPoints
 * 
 * Description: Use the distance formula to get the distance between 2 points
 * 
 * Parameters:
 *     startPoint: 1st point
 *     endPoint: 2nd point
 * 
 * Returns: the distance between the 2 points passed in as arguments
 */
double getDistanceBetweenPoints(
    _In_ CvPoint3D32f startPoint, 
    _In_ CvPoint3D32f endPoint)
{
    double deltaX = endPoint.x - startPoint.x;
    double deltaY = endPoint.y - startPoint.y;
    double deltaZ = endPoint.z - startPoint.z;
    
    return sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
}

/* Function: getMeshFlows
 * 
 * Description: Finds the flow of each point in the mesh by finding the k closest 
 *     mesh points in the sebsequent frame and finding a weighted average of those 
 *     flows, based on their distances to the mesh point.
 * 
 * Parameters:
 *     flows: output Vector array to contain the deltas of the flows
 *     f1MeshPoints: input CvPoint3D32f array of mesh points from frame 1
 *     f1NumMeshPoints: number of mesh points from frame 1
 *     f2MeshPoints: input CvPoint3D32f array of mesh points from frame 2
 *     f2NumMeshPoints: number of mesh points from frame 2
 */
void getMeshFlows(
    _Out_ Vector *flows,
    _In_ CvPoint3D32f *f1MeshPoints,
    _In_ int f1NumMeshPoints, 
    _In_ CvPoint3D32f *f2MeshPoints,
    _In_ int f2NumMeshPoints)
{
    // go through each frame 2 mesh point
    for (int i = 0; i < f2NumMeshPoints; i++)
    {
        CvPoint3D32f f2CurMeshPoint = f2MeshPoints[i];
        
        double shortestDistances[K];
        int shortestDistanceIndices[K];

        // find the mesh point from frame 1 that is closest to the current frame 
        // 2 mesh point k times, excluding ones already found to be closer
        for (int j = 0; j < K; j++)
        {
            int shortestDistanceIndex;
            double shortestDistance = DBL_MAX;
            
            // go through the mesh points from frame 1
            for (int n = 0; n < f1NumMeshPoints; n++)
            {
                bool indexUsed = false;
                
                // make sure that the current frame 1 mesh point isn't already 
                // contained in the list of closest feature points to the frame 
                // 2 mesh point
                for (int m = 0; m < j; m++)
                {
                    if (shortestDistanceIndices[m] == n)
                    {
                        indexUsed = true;
                        break;
                    }
                }
                
                // if current frame 1 mesh point is already contained in list of 
                // closest feature points to the frame 2 mesh point, skip to the 
                // next frame 1 mesh point
                if (indexUsed)
                {
                    continue;
                }
                
                // get the distance from frame 2 mesh point to current frame 1 
                // mesh point
                double curDistance = getDistanceBetweenPoints(f2CurMeshPoint, f1MeshPoints[n]);
                
                // if distance is the shortest distance so far, set it as the 
                // shortest distance, and save the index of the frame 1 mesh point
                if (curDistance < shortestDistance)
                {
                    shortestDistance = curDistance;
                    shortestDistanceIndex = n;
                }
            }
            
            bool insertAtEnd = true;
            
            // insert the closest frame 1 mesh point found in this iteration into 
            // the sorted array of k closest points
            for (int n = 0; n < j; n++)
            {
                if (shortestDistance < shortestDistances[n])
                {
                    for (int m = j; m > n; n--)
                    {
                        shortestDistances[m] = shortestDistances[m-1];
                        shortestDistanceIndices[m] = shortestDistanceIndices[m-1];
                    }
                    
                    shortestDistances[n] = shortestDistance;
                    shortestDistanceIndices[n] = shortestDistanceIndex;
                    
                    insertAtEnd = false;
                    break;
                }
            }
            
            if (insertAtEnd)
            {
                shortestDistances[j] = shortestDistance;
                shortestDistanceIndices[j] = shortestDistanceIndex;
            }
        }
        
        // create an array of weights, where the indices correspond to the indices
        // of the sorted array of closest points
        double weights[K];
        
        // we need to get the sum of the weights to find the percent to multiply 
        // each flow between the mesh point of frames 1 and 2 to to find the 
        // final flow
        double weightSum = 0.0;
        
        for (int j = 0; j < K; j++)
        {
            // set the weight to be the distance of the k-th furthest frame 1 
            // mesh point from the frame 2 mesh point divided by the distance 
            // of the frame 2 mesh point from the frame 1 mesh point
            weights[j] = shortestDistances[K-1]/shortestDistances[j];
            weightSum += weights[j];
        }
        
        double deltaX = 0.0;
        double deltaY = 0.0;
        double deltaZ = 0.0;
        
        for (int j = 0; j < K; j++)
        {
            double weightFraction = weights[j]/weightSum;
            
            // deltas for mesh flow point is determined by the sum of the weight
            // fractions mulitiplied by the flow between the k closest mesh points
            // between frames 1 and 2
            deltaX += weightFraction * (f1MeshPoints[shortestDistanceIndices[j]].x - f2CurMeshPoint.x);
            deltaY += weightFraction * (f1MeshPoints[shortestDistanceIndices[j]].y - f2CurMeshPoint.y);
            deltaZ += weightFraction * (f1MeshPoints[shortestDistanceIndices[j]].z - f2CurMeshPoint.z);
        }
        
        flows[i].deltaX = deltaX;
        flows[i].deltaY = deltaY;
        flows[i].deltaZ = deltaZ;
    }    
}

/* Function: readMeshPointsFromInputFile
 * 
 * Description: reads the 3D wing mesh features outputted by mesh.cpp from file. 
 *     File must be in the format:
 * 
 *     <number of mesh points>
 *     <x coordinate of mesh point 1> <y coordinate of mesh point 1> <z coordinate of mesh point 1>
 *     ...
 *     <x coordinate of mesh point n> <y coordinate of mesh point n> <z coordinate of mesh point n>
 * 
 * Parameters:
 *     meshPoints: CvPoint3D32f array to store the mesh points read in from file
 *     numMeshPoints: number of mesh read in from file
 *     filename: path of the input file containing the mesh points
 * 
 * Returns: 0 on success, error code on error.
 */
int readMeshPointsFromInputFile(
    _Out_ CvPoint3D32f **meshPoints,
    _Out_ int *numMeshPoints,
    _In_ char *filename)
{
    // open the file for reading
    FILE *meshPointsFile = fopen(filename, "r");
    
    if (meshPointsFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }
    
    // get the number of mesh points
    if (fscanf(meshPointsFile, "%d", &(*numMeshPoints)) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    // allocate memory to store the points as CvPoint3D32f
    *meshPoints = (CvPoint3D32f *)malloc(*numMeshPoints * sizeof(CvPoint3D32f));
    
    if (*meshPoints == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    float xBuffer, yBuffer, zBuffer;
    
    // for each point read from file, add to mesh points array
    for (int i = 0; i < *numMeshPoints; i++)
    {
        if (fscanf(meshPointsFile, "%f %f %f", &xBuffer, &yBuffer, &zBuffer) != 3)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
             
        (*meshPoints)[i].x = xBuffer;
        (*meshPoints)[i].y = yBuffer;
        (*meshPoints)[i].z = zBuffer;
    }

    // close the file
    fclose(meshPointsFile);
    
    return 0;
}

/* Function: writeFlowsToFile
 * 
 * Description: writes the mesh points and the x, y, and z deltas of their flows 
 *     to output file, in the format to be read in by MATLAB as an n by 6 array:
 * 
 *     <x coordinate of point 1> <y coordinate of point 1> <z coordinate of point 1> <delta x of point 1 flow> <delta y of point 1 flow> <delta z of point 1 flow>
 *     ...
 *     <x coordinate of point n> <y coordinate of point n> <z coordinate of point n> <delta x of point n flow> <delta y of point n flow> <delta z of point n flow>
 * 
 * Parameters:
 *     flows: Vector array of flows for corresponding mesh points
 *     meshPoints: CvPoint3D32f array of mesh points
 *     numMeshPoints: number of mesh points
 *     outputFilename: path of output file
 * 
 * Returns: 0 on success, error code on error.
 */
int writeFlowsToFile(
    _In_ Vector *flows, 
    _In_ CvPoint3D32f *meshPoints, 
    _In_ int numMeshPoints, 
    char *outputFilename)
{
    // open the output file for writing
    FILE *outputFile = fopen(outputFilename, "w");
    
    if (outputFile == NULL)
    {
        return OUTPUT_FILE_OPEN_ERROR;
    }
    
    // write each mesh point and it corresponding flow to output file
    for (int i = 0; i < numMeshPoints; i++)
    {
        fprintf(outputFile, "%f %f %f ", meshPoints[i].x, meshPoints[i].y, meshPoints[i].z);
        fprintf(outputFile, "%f %f %f\n", flows[i].deltaX, flows[i].deltaY, flows[i].deltaZ);
    }
    
    // close output file
    fclose(outputFile);
    
    return 0;   
}

/* Function: flow
 * 
 * Description: uploads the 3D mesh points of frames 1 and 2 of a wing surface and
 *     finds the flow between frames 1 and 2 for each point in frame 2. Outputs
 *     the calculated flows to file
 * 
 * Parameters:
 *     frame1MeshPointsFilename: filename of the frame 1 mesh points
 *     frame2MeshPointsFilename: filename of the frame 2 mesh points
 *     flowsFilename: filename to write the flows to
 *     errorMessage: string to output an error message to, on error
 * 
 * Returns: 0 on success, 1 on error.
 */
int flow(
    _In_ char *frame1MeshPointsFilename,
    _In_ char *frame2MeshPointsFilename, 
    _In_ char *flowsFilename,
    _Out_ char *errorMessage)
{
    // variable to store error status returned from functions
    int status;
    
    CvPoint3D32f *f1MeshPoints;
    int f1NumMeshPoints;
    
    // read in mesh points from file
    status = readMeshPointsFromInputFile(&f1MeshPoints, &f1NumMeshPoints, frame1MeshPointsFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open frame 1 mesh file.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "frame 1 mesh file has incorrect format.");
        return 1;
    } 
    
    if (status == OUT_OF_MEMORY_ERROR)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    CvPoint3D32f *f2MeshPoints;
    int f2NumMeshPoints;
    
    // read in mesh points from file
    status = readMeshPointsFromInputFile(&f2MeshPoints, &f2NumMeshPoints, frame2MeshPointsFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open frame 2 mesh file.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "frame 2 mesh file has incorrect format.");
        return 1;
    }
    
    if (status == OUT_OF_MEMORY_ERROR)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    Vector *flows = (Vector *)malloc(f2NumMeshPoints * sizeof(Vector));
    
    if (flows == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    // get the estimated flows for each mesh point
    getMeshFlows(flows, f1MeshPoints, f1NumMeshPoints, f2MeshPoints, f2NumMeshPoints);
    
    CvPoint3D32f *flowStartPoints = (CvPoint3D32f *)malloc(f2NumMeshPoints * sizeof(CvPoint3D32f));
    
    for (int i = 0; i < f2NumMeshPoints; i++)
    {
        flowStartPoints[i].x = f2MeshPoints[i].x + flows[i].deltaX;
        flowStartPoints[i].y = f2MeshPoints[i].y + flows[i].deltaY;
        flowStartPoints[i].z = f2MeshPoints[i].z + flows[i].deltaZ;
        
        flows[i].deltaX *= -1;
        flows[i].deltaY *= -1;
        flows[i].deltaZ *= -1;
    }
    
    // write the mesh points and their flows to file
    status = writeFlowsToFile(flows, flowStartPoints, f2NumMeshPoints, flowsFilename);
    
    if (status == OUTPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open output file.");
        return 1;
    }
        
    // cleanup
    free(f1MeshPoints);
    free(f2MeshPoints);
    free(flows);
    free(flowStartPoints);
    
    return 0;
}

