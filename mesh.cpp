/* File: mesh.cpp
 *
 * Description: This file contains the program to upload the 3D triangulated points
 *     from outputted from triangulation.cpp, the vertex points of the contours 
 *     of the region(s) of interest selected in features.cpp, and the camera 
 *     coefficients. The triangulated points are first converted into PCA space.
 *     Then we calculate the thin plate spline for the points, returning a regular
 *     x,y grid with interpolated z coordinates. Then we convert the grid points 
 *     from PCA space back to the original space. We calculate the ideal 2D u,v
 *     coordinates of the grid points, and only keep the ones that lie within the
 *     inputted contour(s). We then output these valid points to file.
 *
 * Author: Ming Guo
 * Created: 8/24/11
 */

#include "wing.h"
#include "mesh.h"
#include "features.h"
#include "triangulation.h"

// struct to store values for PCA calculation
typedef struct
{
    int dimension;
    gsl_matrix *covarianceMatrix;
    gsl_eigen_symmv_workspace *workspace;
    gsl_vector *eigenvalues;
    gsl_matrix *eigenvectors;
    gsl_vector *mean;
    gsl_matrix *meanSubstractedPoints;
} Data;

/* Function: TPSBase
 * 
 * Description: Used for TPS calculations
 * 
 * Parameters:
 *     r: edge length to calculate the TPS base for
 * 
 * Returns: result of the calculation
 */
double TPSBase(_In_ double r)
{
    if (r == 0.0)
    {
        return 0.0;
    }
    else
    {
        return r * r * log(r);
    }
}

/* Function: TPS
 * 
 * Description: Calculate Thin Plate Spline (TPS) weights from control points 
 *     and build a new height grid by interpolating with them. Code taken and
 *     modified from http://elonen.iki.fi/code/tpsdemo/index.html
 * 
 * Parameters:
 *     features3DPrime: feature points to serve as control for TPS calculation
 *     numPoints: number of feature points
 *     grid: 2D double array to store output of the TPS calculations. The rows 
 *         represent x coordinate values, columns represent y coordinate values, 
 *         and values of the array represent the z coordinate values
 *     gridHeight: height of the grid
 *     gridWidth: width of the grid
 *     gridHeightStartIndex: The grid row and column indices start at 0 because
 *         it is an array. However, the actual x and y start values do not
 *         necessarily start 0, so gridHeightStartIndex represents the value at
 *         which the y coordinates start at in the grid.
 *     gridWidthStartIndex: the value at which the x coordinates start at in the
 *         grid
 *     regularization: smoothing parameter that is the ratio of the error variance
 *         to the scale parameter of the covariance function
 * 
 * Returns: 0 on success, error code on error.
 */
int TPS(
    _In_ CvPoint3D32f *features3DPrime,
    _In_ int numPoints,
    _Out_ double **grid,
    _In_ int gridHeight,
    _In_ int gridWidth,
    _In_ int gridHeightStartIndex,
    _In_ int gridWidthStartIndex,
    _In_ double regularization)
{    
    // We need at least 3 points to define a plane
    if (numPoints < 3)
    {
        return NOT_ENOUGH_POINTS_ERROR;
    }
    
    // Allocate the matrix and vector
    gsl_matrix *L = gsl_matrix_alloc(numPoints+3, numPoints+3);
    gsl_vector *V = gsl_vector_alloc(numPoints+3);
    
    if (L == NULL || V == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // Fill K (p x p, upper left of L) and calculate mean edge length from control 
    // points. K is symmetrical so we really have to calculate only about half 
    // of the coefficients.
    double a = 0.0;
    
    for (int i = 0; i < numPoints; i++)
    {
        CvPoint3D32f iPoint = features3DPrime[i];
        
        for (int j = i+1; j < numPoints; j++)
        {
            CvPoint3D32f jPoint = features3DPrime[j];
            
            iPoint.z = 0.0;
            jPoint.z = 0.0;
                                    
            double edgeLength = sqrt(pow(iPoint.x-jPoint.x, 2) + pow(iPoint.y-jPoint.y, 2));
            
            gsl_matrix_set(L, i, j, TPSBase(edgeLength));
            gsl_matrix_set(L, j, i, TPSBase(edgeLength));
            
            a += edgeLength * 2;
        }
    }
    
    a /= (double)(numPoints * numPoints);
    
    // Fill the rest of L
    for (int i = 0; i < numPoints; i++)
    {
        // diagonal: reqularization parameters (lambda * a^2)
        gsl_matrix_set(L, i, i, regularization * a * a);
        
        // P (p x 3, upper right)
        gsl_matrix_set(L, i, numPoints, 1.0);
        gsl_matrix_set(L, i, numPoints+1, features3DPrime[i].x);
        gsl_matrix_set(L, i, numPoints+2, features3DPrime[i].y);
        
        // P transposed (3 x p, bottom left)
        gsl_matrix_set(L, numPoints, i, 1.0);
        gsl_matrix_set(L, numPoints+1, i, features3DPrime[i].x);
        gsl_matrix_set(L, numPoints+2, i, features3DPrime[i].y);
    }
    
    // O (3 x 3, lower right)
    for (int i = numPoints; i < numPoints+3; i++)
    {
        for (int j = numPoints; j < numPoints+3; j++)
        {
            gsl_matrix_set(L, i, j, 0.0);
        }
    }
        
    // Fill the right hand vector V
    for (int i = 0; i < numPoints; i++)
    {
        gsl_vector_set(V, i, features3DPrime[i].z);
    }
    
    gsl_vector_set(V, numPoints, 0.0);
    gsl_vector_set(V, numPoints+1, 0.0);
    gsl_vector_set(V, numPoints+2, 0.0);
    
    int signum;
    gsl_permutation *perm = gsl_permutation_alloc(numPoints+3);
    
    if (perm == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // Solve the linear system inplace
    gsl_linalg_LU_decomp(L, perm, &signum);
    gsl_linalg_LU_svx(L, perm, V);    
    
    gsl_permutation_free(perm);
    
    // Interpolate grid heights
    for (int x = gridWidthStartIndex; x < gridWidthStartIndex + gridWidth; x++)
    {
        for (int y = gridHeightStartIndex; y < gridHeightStartIndex + gridHeight; y++ )
        {
            double h = gsl_vector_get(V, numPoints) + gsl_vector_get(V, numPoints+1)*x + gsl_vector_get(V, numPoints+2)*y;

            for (int i = 0; i < numPoints; i++)
            {
                CvPoint3D32f iPoint = features3DPrime[i];
                CvPoint3D32f curPoint = cvPoint3D32f((double)x, (double)y, 0.0);

                iPoint.z = 0;
                h += gsl_vector_get(V, i) * TPSBase(sqrt(pow(iPoint.x-curPoint.x, 2) + pow(iPoint.y-curPoint.y, 2) + pow(iPoint.z-curPoint.z, 2)));
            }
            
            grid[x - gridWidthStartIndex][y - gridHeightStartIndex] = h;
        }
    }
    
    gsl_matrix_free(L);
    gsl_vector_free(V);
    
    return 0;
}

/* Function: createPCA
 * 
 * Description: allocate memory for matrices and vectors used for PCA calculation
 * 
 * Parameters:
 *     data: PCA data structure to allocate memory for
 *     dimension: the number of dimensions of the points to convert to PCA space
 *     numPoints: number of points
 * 
 * Returns: 0 on success, error code on error.
 */
int createPCA(
    _Out_ Data *data, 
    _In_ int dimension, 
    _In_ int numPoints)
{
    data->dimension = dimension;
    data->covarianceMatrix = gsl_matrix_alloc(dimension, dimension);
    data->eigenvalues = gsl_vector_alloc(dimension);
    data->eigenvectors = gsl_matrix_alloc(dimension, dimension);
    data->mean = gsl_vector_alloc(dimension);
    data->meanSubstractedPoints = gsl_matrix_alloc(numPoints, dimension);
    data->workspace = gsl_eigen_symmv_alloc(dimension);
    
    if (!data->covarianceMatrix ||
        !data->eigenvalues ||
        !data->eigenvectors ||
        !data->mean ||
        !data->meanSubstractedPoints ||
        !data->workspace)
    {
        return OUT_OF_MEMORY_ERROR;
    }
        
    return 0;
}

/* Function: destroyPCA
 * 
 * Description: de-allocate memory for matrices and vectors used for PCA calculation
 * 
 * Parameters:
 *     data: PCA data structure to de-allocate memory for
 */
void destroyPCA(_InOut_ Data *data)
{
    gsl_eigen_symmv_free(data->workspace); 
    gsl_matrix_free(data->meanSubstractedPoints);
    gsl_vector_free(data->mean);
    gsl_matrix_free(data->eigenvectors);
    gsl_vector_free(data->eigenvalues);
    gsl_matrix_free(data->covarianceMatrix);
}

/* Function: reversePCA
 * 
 * Description: Convert points PCA space back to the original space by reversing 
 *     the PCA calculations on them.
 * 
 * Parameters:
 *     features3D: output array of CvPoint3D32f containing the 3D points in their
 *         original space
 *     features3DPrime: input array of CvPoint3D32f containing the 3D points in
 *         PCA space
 *     numFeatures: number of features
 *     PCAData: the matrices and vectors calculated in the conversion to PCA space,
 *         used for the reverse PCA calculations
 */
int reversePCA(
    _Out_ CvPoint3D32f *features3D,
    _In_ CvPoint3D32f *features3DPrime,
    _In_ int numFeatures,
    _In_ Data *PCAData)
{    
    // allocate a matrix to store the current points in PCA space
    gsl_matrix *scores = gsl_matrix_alloc(numFeatures, 3);
    
    if (scores == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // fill the matrix "scores" with the grid points
    for (int i = 0; i < numFeatures; i++)
    {
        gsl_matrix_set(scores, i, 0, features3DPrime[i].x);
        gsl_matrix_set(scores, i, 1, features3DPrime[i].y);
        gsl_matrix_set(scores, i, 2, features3DPrime[i].z);
    }
    
    int signum;
    
    // Define and allocate all the used matrices
    gsl_matrix *inverseEigenvectors = gsl_matrix_alloc(3, 3);
    gsl_permutation *perm = gsl_permutation_alloc(3);
    
    if (inverseEigenvectors == NULL || perm == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
        
    // Get LU decomposition of PCAData->eigenvectors to get its inverse
    gsl_linalg_LU_decomp(PCAData->eigenvectors, perm, &signum);
    
    // Invert the matrix PCAData->eigenvectors
    gsl_linalg_LU_invert(PCAData->eigenvectors, perm, inverseEigenvectors);
    
    // allocate points matrix to store matrix multiplication result
    gsl_matrix *points = gsl_matrix_alloc(numFeatures, 3);
    
    if (points == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // multiply the inverse of the eigenvectors by scores
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, scores, inverseEigenvectors, 0.0, points);
        
    // add back the mean for each dimension to each point to get the point in the 
    // original space
    for (int i = 0; i < numFeatures; i++)
    {   
        features3D[i].x = gsl_matrix_get(points, i, 0) + gsl_vector_get(PCAData->mean, 0);
        features3D[i].y = gsl_matrix_get(points, i, 1) + gsl_vector_get(PCAData->mean, 1);
        features3D[i].z = gsl_matrix_get(points, i, 2) + gsl_vector_get(PCAData->mean, 2);
    }
    
    // cleanup
    gsl_matrix_free(scores);
    gsl_matrix_free(inverseEigenvectors);
    gsl_permutation_free(perm);
    gsl_matrix_free(points);
    
    return 0;
}

/* Function: PCA
 * 
 * Description: Perform PCA calculations and store the results in a CvPoint3D32f 
 *     array
 * 
 * Parameters:
 *     features3DPrime: CvPoint3D32f array to store the resulting points of the
 *         PCA calculation, in PCA space
 *     features3D: the features points to convert to PCA space
 *     numFeatures: number of features
 *     PCAData: the matrices and vectors calculated during PCA, saved for later
 *         reversing the calculations
 */
int PCA(
    _Out_ CvPoint3D32f *features3DPrime, 
    _In_ CvPoint3D32f *features3D, 
    _In_ int numFeatures,
    _Out_ Data *PCAData)
{   
    double xSum = 0.0;
    double ySum = 0.0;
    double zSum = 0.0;
    
    // Find mean of each of the dimensions, and store in vector data->mean
    for (int i = 0; i < numFeatures; i++)
    {
        xSum += features3D[i].x;
        ySum += features3D[i].y;
        zSum += features3D[i].z;        
    }
    
    gsl_vector_set(PCAData->mean, 0, xSum/(double)numFeatures);
    gsl_vector_set(PCAData->mean, 1, ySum/(double)numFeatures);
    gsl_vector_set(PCAData->mean, 2, zSum/(double)numFeatures);
    
    // Get mean-substracted data into matrix data->meanSubstractedPoints.
    for (int i = 0; i < numFeatures; i++)
    {
        double meanSubtractedXValue = features3D[i].x - gsl_vector_get(PCAData->mean, 0);
        double meanSubtractedYValue = features3D[i].y - gsl_vector_get(PCAData->mean, 1);
        double meanSubtractedZValue = features3D[i].z - gsl_vector_get(PCAData->mean, 2);
        
        gsl_matrix_set(PCAData->meanSubstractedPoints, i, 0, meanSubtractedXValue);
        gsl_matrix_set(PCAData->meanSubstractedPoints, i, 1, meanSubtractedYValue);
        gsl_matrix_set(PCAData->meanSubstractedPoints, i, 2, meanSubtractedZValue);

    }        
        
    // Compute Covariance matrix
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0/(double)numFeatures, PCAData->meanSubstractedPoints, PCAData->meanSubstractedPoints, 0.0, PCAData->covarianceMatrix);
        
    // Get eigenvectors, sort by eigenvalue.
    gsl_eigen_symmv(PCAData->covarianceMatrix, PCAData->eigenvalues, PCAData->eigenvectors, PCAData->workspace);
    gsl_eigen_symmv_sort(PCAData->eigenvalues, PCAData->eigenvectors, GSL_EIGEN_SORT_ABS_DESC);
    
    double maxAbsVals[3] = {0.0, 0.0, 0.0};
    
    // get the maximum absolute value of each column
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            double val = gsl_matrix_get(PCAData->eigenvectors, i, j);
            
            if (fabs(val) > fabs(maxAbsVals[j]))
            {
                maxAbsVals[j] = val;
            }
        }
    }
    
    // If the maximum absolute value of a column is negative, multiply everything
    // in that column by -1. Why? No idea, but that's how the MATLAB PCA function 
    // does it.
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {            
            if (maxAbsVals[j] < 0)
            {
                double val = gsl_matrix_get(PCAData->eigenvectors, i, j);
                gsl_matrix_set(PCAData->eigenvectors, i, j, -1 * val);
            }
        }
    }    
    
    gsl_matrix *scores = gsl_matrix_alloc(numFeatures, 3);
    
    if (scores == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // multiply the eigenvectors from the PCA by the feature points with the mean
    // subtracted to get the points in PCA space
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, PCAData->meanSubstractedPoints, PCAData->eigenvectors, 0.0, scores);
    
    for (int i = 0; i < numFeatures; i++)
    {
        features3DPrime[i].x = gsl_matrix_get(scores, i, 0);
        features3DPrime[i].y = gsl_matrix_get(scores, i, 1);
        features3DPrime[i].z = gsl_matrix_get(scores, i, 2);
    }
    
    // cleanup
    gsl_matrix_free(scores);
    
    return 0;
}

/* Function: getXPrimeYPrimeMaxMin
 * 
 * Description: get the max and min values of x and y values of the features points
 *     in PCA space, used for setting up correct grid start index values
 * 
 * Parameters:
 *     features3DPrime: CvPoint3D32f array containing feature points in PCA space
 *     numFeatures: number of features
 *     xPrimeMax: max x value of features3DPrime, found by function
 *     yPrimeMax: max y value of features3DPrime, found by function
 *     xPrimeMin: min x value of features3DPrime, found by function
 *     yPrimeMin: min y value of features3DPrime, found by function
 */
void getXPrimeYPrimeMaxMin(
    _In_ CvPoint3D32f *features3DPrime,
    _In_ int numFeatures,
    _Out_ double *xPrimeMax,
    _Out_ double *yPrimeMax,
    _Out_ double *xPrimeMin,
    _Out_ double *yPrimeMin)
{
    // set initial values for min, max values of x, y
    *xPrimeMax = DBL_MIN;
    *yPrimeMax = DBL_MIN;
    *xPrimeMin = DBL_MAX;
    *yPrimeMin = DBL_MAX;
    
    for (int i = 0; i < numFeatures; i++)
    {
        CvPoint3D32f curPoint = features3DPrime[i];
        
        if (curPoint.x > *xPrimeMax)
        {
            *xPrimeMax = curPoint.x;
        }
        if (curPoint.x < *xPrimeMin)
        {
            *xPrimeMin = curPoint.x;
        }
        if (curPoint.y > *yPrimeMax)
        {
            *yPrimeMax = curPoint.y;
        }
        if (curPoint.y < *yPrimeMin)
        {
            *yPrimeMin = curPoint.y;
        }
    }
}

/* Function: calculateIdealFeatures
 * 
 * Description: calculates the re-projected ideal 2D features of the triangulated
 *     3D coordinates based on the 11 camera coefficients.
 *
 * Parameters:
 *     idealFeatures2D: array to store the ideal re-projected 2D features
 *     features3D: array of triangulated 3D features
 *     numFeatures: number of features
 *     cameraCoefficients: array of the 11 coefficients of the camera to re-project
 *         the 3D features for.
 */
void calculateIdealFeatures(
    _Out_ CvPoint2D32f *idealFeatures2D,
    _In_ CvPoint3D32f *features3D, 
    _In_ int numFeatures, 
    _In_ double *cameraCoefficients)
{
    // for each feature, calculate the ideal u,v coordinates from the triangulated 
    // 3D point for a particular camera based on the DLT method, and store the 
    // result in the idealFeatures2D array
    for (int i = 0; i < numFeatures; i++)
    {
        idealFeatures2D[i].x = (features3D[i].x * cameraCoefficients[0] + 
                                features3D[i].y * cameraCoefficients[1] + 
                                features3D[i].z * cameraCoefficients[2] + 
                                cameraCoefficients[3]) /
                               (features3D[i].x * cameraCoefficients[8] +
                                features3D[i].y * cameraCoefficients[9] +
                                features3D[i].z * cameraCoefficients[10] + 1);
                                   
        idealFeatures2D[i].y = (features3D[i].x * cameraCoefficients[4] + 
                                features3D[i].y * cameraCoefficients[5] + 
                                features3D[i].z * cameraCoefficients[6] + 
                                cameraCoefficients[7]) /
                               (features3D[i].x * cameraCoefficients[8] +
                                features3D[i].y * cameraCoefficients[9] +
                                features3D[i].z * cameraCoefficients[10] + 1);
    }
}

/* Function: readContourVerticesFromInputFile
 * 
 * Description: reads vertices of contour(s) selected in features.cpp from file
 *     and creates a CvSeq for the contour(s). File must be in the format:
 * 
 *     <number of contours>
 *     <contour 1 name> <number of vertices in contour 1>
 *     <x coordinate of contour 1 vertex 1> <y coordinate of contour 1 vertex 1>
 *     ...
 *     <x coordinate of contour 1 vertex n> <y coordinate of contour 1 vertex n>
 *     ...
 *     <contour m name> <number of vertices in contour m> 
 *     <x coordinate of contour m vertex 1> <y coordinate of contour m vertex 1>
 *     ...
 *     <x coordinate of contour m vertex n> <y coordinate of contour m vertex n>
 * 
 * Parameters:
 *     contours: CvSeq to store the contours
 *     contourStorage: memory storage for CvSeq
 *     numContours: number of contours
 *     filename: name of the input file
 * 
 * Returns: 0 on success, error code on error.
 */
int readContourVerticesFromInputFile(
    _Out_ CvSeq **contours, 
    _Out_ CvMemStorage *contourStorage, 
    _Out_ int *numContours,
    _In_ char *filename)
{    
    // open the file for reading
    FILE *contourVerticesFile = fopen(filename, "r");
    
    if (contourVerticesFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }
    
    char stringBuffer[50];
    int xBuffer, yBuffer;
    
    // get the number of contours
    if (fscanf(contourVerticesFile, "%d", &(*numContours)) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    // create a CvSeq for the first contour
    *contours = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), contourStorage);
    
    // for each contour, get the number of vertices and name of the countrour,
    // as well as each vertex point coordinates
    for (int i = 0; i < *numContours; i++)
    {
        int numVertices;
        char contourNameBuffer[MAX_CONTOUR_NAME_LENGTH];
        
        if (fscanf(contourVerticesFile, "%s %d", contourNameBuffer, &numVertices) != 2)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        // read in the contour vertices and push into the current CvSeq
        for (int j = 0; j < numVertices; j++)
        {            
            if (fscanf(contourVerticesFile, "%d %d", &xBuffer, &yBuffer) != 2)
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
            
            CvPoint newPoint;
            newPoint.x = xBuffer;
            newPoint.y = yBuffer;
            
            cvSeqPush(*contours, &newPoint);
        }
        
        // if there are more contours, create a new CvSeq and set the h_next 
        // pointer of the current contour to it
        if (i < *numContours-1)
        {
            (*contours)->h_next = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), contourStorage);
            (*contours)->h_next->h_prev = *contours;
            
            *contours = (*contours)->h_next;
        }
    }
    
    // close the input file
    fclose(contourVerticesFile);
    
    // "rewind" the contours linked list so that "contours" points at the first 
    // contour
    while ((*contours)->h_prev)
    {
        *contours = (*contours)->h_prev;
    }
    
    return 0;
}

/* Function: read3DFeaturesFromInputFile
 * 
 * Description: reads the triangulated 3D features outputted by triangulation.cpp
 *     from file. File must be in the format:
 * 
 *     <number of contours>
 *     <contour 1 name> <number of points in contour 1>
 *     <1 or 0 indicating whether contour 1 feature 1 is valid> <x coordinate of contour 1 feature 1> <y coordinate of contour 1 feature 1> <z coordinate of contour 1 feature 1>
 *     ...
 *     <1 or 0 indicating whether contour 1 feature n is valid> <x coordinate of contour 1 feature n> <y coordinate of contour 1 feature n> <z coordinate of contour 1 feature n>
 *     ...
 *     <contour m name> <number of points in contour m>
 *     <1 or 0 indicating whether contour m feature 1 is valid> <x coordinate of contour m feature 1> <y coordinate of contour m feature 1> <z coordinate of contour m feature 1>
 *     ...
 *     <1 or 0 indicating whether contour m feature n is valid> <x coordinate of contour m feature n> <y coordinate of contour m feature n> <z coordinate of contour m feature n>
 * 
 * Parameters:
 *     features3D: CvPoint3D32f array to store the valid points for each contour
 *     numFeaturesInContours: array of ints containing number of valid features 
 *         for each contour
 *     numContours: number of contours
 *     filename: path of the input file containing the 3D features
 * 
 * Returns: 0 on success, error code on error.
 */
int read3DFeaturesFromInputFile(
    _Out_ CvPoint3D32f ***features3D,
    _Out_ int **numFeaturesInContours,
    _Out_ int *numContours,
    _In_ char *filename)
{
    // open the file for reading
    FILE *featuresFile = fopen(filename, "r");
    
    if (featuresFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }    
    
    // get the number of features, including invalid features
    if (fscanf(featuresFile, "%d", &(*numContours)) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    if (*numContours < 1)
    {
        return INVALID_NUM_CONTOURS_ERROR;
    }
    
    // allocate memory to store the features as CvPoint3D32f
    *features3D = (CvPoint3D32f **)malloc((*numContours) * sizeof(CvPoint3D32f *));
    *numFeaturesInContours = (int *)malloc((*numContours) * sizeof(int));
    
    if (*features3D == NULL || *numFeaturesInContours == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    char contourNameBuffer[MAX_CONTOUR_NAME_LENGTH];
    char flagBuffer[1];
    float xBuffer, yBuffer, zBuffer;
    
    // for each contour, get the coordinates of the 3D feature points
    for (int i = 0; i < *numContours; i++)
    {
        if (fscanf(featuresFile, "%s %d", contourNameBuffer, &((*numFeaturesInContours)[i])) != 2)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        (*features3D)[i] = (CvPoint3D32f *)malloc((*numFeaturesInContours)[i] * sizeof(CvPoint3D32f));
        
        if ((*features3D)[i] == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
        
        int numValidFeatures = 0;
        
        // for each feature in file, add to features3D array if valid
        for (int j = 0; j < (*numFeaturesInContours)[i]; j++)
        {
            if (fscanf(featuresFile, "%s %f %f %f", flagBuffer, &xBuffer, &yBuffer, &zBuffer) != 4)
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
            
            if (flagBuffer[0] == '1')
            {                 
                (*features3D)[i][numValidFeatures].x = xBuffer;
                (*features3D)[i][numValidFeatures].y = yBuffer;
                (*features3D)[i][numValidFeatures].z = zBuffer;            
                
                numValidFeatures++;
            }
            else if (flagBuffer[0] == '0')
            {
                continue;
            }
            else
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
        }
        
        (*numFeaturesInContours)[i] = numValidFeatures;
    }

    // close the file
    fclose(featuresFile);
    
    return 0;
}

/* Function: writeGridPointsToFile
 * 
 * Description: writes the grid points representing the wing mesh to output file,
 *     in the format:
 * 
 *     <number of valid points in wing mesh>
 *     <x coordinate of point 1> <y coordinate of point 1> <z coordinate of point 1>
 *     ...
 *     <x coordinate of point n> <y coordinate of point n> <z coordinate of point n>
 * 
 * Parameters:
 *     meshPointsFilenames: names of output files for the meshes of each contour
 *     features3DGrid: CvPoint3D32f arrays containing all points in mesh for each
 *         contour
 *     numGridPoints: number of grid points for each contour (containing valid and
 *         invalid points)
 *     validFeatureIndicator: char array indicating whether point in grid is part
 *         of the wing, as specified by the input contours, for each contour
 *     numGridPointsInContours: int array containing the number of valid mesh points 
 *         in each contour
 *     numContours: number of contours
 * 
 * Returns: 0 on success, error code on error.
 */
int writeGridPointsToFile(
    _In_ char **meshPointsFilenames,
    _In_ CvPoint3D32f **features3DGrid, 
    _In_ int *numGridPoints,
    _In_ char **validFeatureIndicator,
    _In_ int *numGridPointsInContours,
    _In_ int numContours)
{
    // for each contour, write the mesh points to a different file
    for (int i = 0; i < numContours; i++)
    {        
        // open the output file for writing
        FILE *outputFile = fopen(meshPointsFilenames[i], "w");
        
        if (outputFile == NULL)
        {
            return OUTPUT_FILE_OPEN_ERROR;
        }
        
        // output number of valid points on the wing
        fprintf(outputFile, "%d\n", numGridPointsInContours[i]);
        
        // go through each grid point and output to file the ones that are contained
        // within the contour, as indicated by validFeatureIndicator
        for (int j = 0; j < numGridPoints[i]; j++)
        {        
            if (validFeatureIndicator[i][j])
            {
                fprintf(outputFile, "%f %f %f\n", features3DGrid[i][j].x, features3DGrid[i][j].y, features3DGrid[i][j].z);
            }
        }
        
        // close output file
        fclose(outputFile);
    }
    
    return 0;   
}

/* Function: mesh
 * 
 * Description: Reads in 3D feature points, contours, and camera coefficients 
 *     from file. Converts the 3D feature points to PCA space, creates a thin
 *     plate spline over a 3D grid, and converts the mesh back to the original
 *     space. Using the camera coefficients to project the points of the mesh
 *     to 2D, we then set only points that lie within the input contours as
 *     valid. The resulting valid mesh points are then outputed to file for each
 *     input contour.
 * 
 * Parameters:
 *     features3DFilename: filename of the 3D features
 *     contoursFilename: filename of the vertices for each contour
 *     cameraCoefficientsFilename: filename of the camera coefficients
 *     meshPointsFilenames: filenames of the files to write each contour mesh
 *         points to
 *     numMeshFiles: number of mesh point files (must be same as number of contours)
 *     regularization: smoothing parameter for the TPS calculations
 *     errorMessage: string to output an error message to, on error
 * 
 * Returns: 0 on success, 1 on error.
 */
int mesh(
    _In_ char *features3DFilename,
    _In_ char *contoursFilename, 
    _In_ char *cameraCoefficientsFilename,
    _In_ char **meshPointsFilenames,
    _In_ int numMeshFiles,
    _In_ double regularization,
    _Out_ char *errorMessage)
{    
    // variable to store error status returned from functions
    int status;
    
    int numContours;
    CvPoint3D32f **features3D;    
    int *numFeaturesInContours;
    
    // read the input triangulated 3D features from file
    status = read3DFeaturesFromInputFile(&features3D, &numFeaturesInContours, &numContours, features3DFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open 3D features file.");
        return 1;
    }
    
    if (status == INVALID_NUM_CONTOURS_ERROR)
    {
        sprintf(errorMessage, "At least 1 contour region required.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "3D features file has incorrect format.");
        return 1;
    } 
    
    if (status == OUT_OF_MEMORY_ERROR)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    if (numContours != numMeshFiles)
    {
        sprintf(errorMessage, "Number of contours passed into function and read in from 3D features file must match.");
        return 1;
    }
    
    CvSeq* contours;
    CvMemStorage *contourStorage = cvCreateMemStorage(0);
        
    if (contourStorage == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    int numContoursFromContourFile;
    
    // read the input region of interest contours from file
    status = readContourVerticesFromInputFile(&contours, contourStorage, &numContoursFromContourFile, contoursFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open contour vertices file.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "Contour vertices file has incorrect format.");
        return 1;
    }
    
    if (numContours != numContoursFromContourFile)
    {
        sprintf(errorMessage, "Number of contours in contour vertices file and 3D features file must match.");
        return 1;
    }
    
    double **cameraCoefficients;
    int numCameras;
    
    // get the number of cameras and 11 camera coefficients for each camera from 
    // file
    status = readCoefficientsFromInputFile(&cameraCoefficients, &numCameras, cameraCoefficientsFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open camera coefficients file.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "Camera coefficients file has incorrect format.");
        return 1;
    }
    
    if (status == INCORRECT_NUM_CAMERAS_ERROR)
    {
        sprintf(errorMessage, "At least 2 cameras are required for triangulation.");
        return 1;
    }    
    
    if (status == OUT_OF_MEMORY_ERROR)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    CvPoint3D32f **features3DGrid = (CvPoint3D32f **)malloc(numContours * sizeof(CvPoint3D32f *));
    char **validFeatureIndicator = (char **)malloc(numContours * sizeof(char *));    
    
    if (features3DGrid == NULL || validFeatureIndicator == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    int *numGridPoints = (int *)malloc(numContours * sizeof(int));
    int *numGridPointsInContours = (int *)malloc(numContours * sizeof(int));

    if (numGridPoints == NULL || numGridPointsInContours == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    CvSeq *contour = contours;
    int k = 0;
    
    // for each contour, calculate the TPS for the feature points
    while (contour)
    {        
        CvPoint3D32f *features3DPrime = (CvPoint3D32f *)malloc(numFeaturesInContours[k] * sizeof(CvPoint3D32f));    
        Data PCAData;
        
        if (createPCA(&PCAData, 3, numFeaturesInContours[k]) == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        // convert the input points to PCA space, and store in features3DPrime
        status = PCA(features3DPrime, features3D[k], numFeaturesInContours[k], &PCAData);
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        double xPrimeMax, yPrimeMax, xPrimeMin, yPrimeMin;
        
        // get the max and min x and y values of features3DPrime, so that we can create
        // a grid of appropriate size that will encompass the features
        getXPrimeYPrimeMaxMin(features3DPrime, numFeaturesInContours[k], &xPrimeMax, &yPrimeMax, &xPrimeMin, &yPrimeMin);
        
        // multiply the ranges of x and y by 3 to create grid size
        int gridWidth = (int) ((xPrimeMax - xPrimeMin) * 3);
        int gridHeight = (int) ((yPrimeMax - yPrimeMin) * 3);
        
        double **grid = (double **)malloc(gridWidth * sizeof(double *));
        
        if (grid == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        for (int i = 0; i < gridWidth; i++)
        {
            grid[i] = (double *)malloc(gridHeight * sizeof(double));
            
            if (grid[i] == NULL)
            {
                sprintf(errorMessage, "Out of memory error.");
                return 1;
            }
        }
        
        // since the grid row and column indices start at 0 because it is an array,
        // but the actual x and y start values do not necessarily start 0, we find
        // the actual start values of x and y. This is the middle point between max
        // and min of x or y, minus half the width or height of the grid
        int gridWidthStartIndex = (int)(((xPrimeMax + xPrimeMin)/2) - (gridWidth/2));
        int gridHeightStartIndex = (int)(((yPrimeMax + yPrimeMin)/2) - (gridHeight/2));
        
        // perform thin plate spline calculations to find the smoothed z coordinates
        // for every point in the grid
        status = TPS(features3DPrime, numFeaturesInContours[k], grid, gridHeight, gridWidth, gridHeightStartIndex, gridWidthStartIndex, regularization);
        
        if (status == NOT_ENOUGH_POINTS_ERROR)
        {
            sprintf(errorMessage, "At least 3 valid feature points are required to define a plane for thin sheet spline function.");
            return 1;
        }
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        numGridPoints[k] = gridHeight * gridWidth;
        
        CvPoint3D32f *features3DGridPrime = (CvPoint3D32f *)malloc(numGridPoints[k] * sizeof(CvPoint3D32f));
        features3DGrid[k] = (CvPoint3D32f *)malloc(numGridPoints[k] * sizeof(CvPoint3D32f));
        
        if (features3DGridPrime == NULL || features3DGrid[k] == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        // transform the grid to CvPoint3D32f points
        for (int i = 0; i < gridWidth; i++)
        {
            for (int j = 0; j < gridHeight; j++)
            {            
                int curIndex = gridWidth*j + i;
                
                features3DGridPrime[curIndex].x = (double) i+gridWidthStartIndex;
                features3DGridPrime[curIndex].y = (double) j+gridHeightStartIndex;
                features3DGridPrime[curIndex].z = grid[i][j];
            }
        }
        
        // convert the grid points from PCA space back to original space, and store
        // the points in CvPoint3D32f array features3DGrid
        status = reversePCA(features3DGrid[k], features3DGridPrime, numGridPoints[k], &PCAData);
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1; 
        }
        
        CvPoint2D32f *idealFeatures2D = (CvPoint2D32f *)malloc(numGridPoints[k] * sizeof(CvPoint2D32f));
        
        if (idealFeatures2D == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }    
        
        // project the 3d grid points to 2D space
        calculateIdealFeatures(idealFeatures2D, features3DGrid[k], numGridPoints[k], cameraCoefficients[0]);
        
        validFeatureIndicator[k] = (char *)malloc(numGridPoints[k] * sizeof(char));
        
        if (validFeatureIndicator[k] == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        // test each 2d grid point to see if it lies within the input contours, and
        // set the correspoinding values of validFeatureIndicator accordingly
        numGridPointsInContours[k] = areFeaturesInContour(idealFeatures2D, numGridPoints[k], validFeatureIndicator[k], contour);
        
        contour = (CvSeq *)(contour->h_next);
        k++;
        
        free(features3DPrime);
        destroyPCA(&PCAData);
    
        for (int i = 0; i < gridWidth; i++)
        {
            free(grid[i]);
        }
    
        free(grid);
        free(features3DGridPrime);
        free(idealFeatures2D);        
    }    
    
    // print the valid 3D mesh points to files for each contour
    writeGridPointsToFile(meshPointsFilenames, features3DGrid, numGridPoints, validFeatureIndicator, numGridPointsInContours, numContours);     
    
    // cleanup
    for (int i = 0; i < numContours; i++)
    {
        free(features3D[i]);
        free(features3DGrid[i]);
        free(validFeatureIndicator[i]);
    }
    
    free(features3D);    
    free(features3DGrid);    
    free(validFeatureIndicator);
    free(numGridPoints);
    free(numGridPointsInContours);
    
    cvReleaseMemStorage(&contourStorage);
    
    for (int i = 0; i < numCameras; i++)
    {
        free(cameraCoefficients[i]);
    }
    
    free(cameraCoefficients);
               
    return 0;
}

