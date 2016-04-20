/* File: triangulation.cpp
 *
 * Description: This file contains the program to upload the 2D feature points from
 *     of n cameras, as well as the 11 camera coefficients of the n cameras used 
 *     for triangulation. Triangulation is then performed for the 2D features and 
 *     the resulting 3D coordinates for the features is written to an output file.
 *
 * Author: Ming Guo
 * Created: 7/27/11
 */

#include "wing.h"
#include "triangulation.h"
#include "features.h"

#define FLOW_LENGTH 1
#define FLOW_ANGLE 0

// calculated from 800*600 / 1.25
#define FACTOR 384000

/* Function: triangulate
 * 
 * Description: Triangulate the 3D coordinates from the 2D coordinates of n cameras
 *     though the 11-parameter DLT method. Get the root mean square error of the
 *     triangulated point and set the point as invalid if it's higher than the
 *     error threshold.
 * 
 * Parameters:
 *     features2D: 2D array of 2D feature points from n cameras
 *     numFeatures: number of features to triangulate. Same throughout all cameras
 *     cameraCoefficients: 2D array of the 11 camera coefficients for n cameras 
 *         used for DLT triangulation
 *     numCameras: number of cameras
 *     features3D: array of triangulated 3D feature points
 *     validFeatureIndicator: Char array where an element has been set to 1 if the
 *         corresponding feature is valid, 0 if the error is above the error 
 *         threshold.
 *     errorThreshold: if the triangulation leaves a residue above the error
 *         threshold, it is set as invalid
 * 
 * Returns: 0 on success, error code on error.
 */
int triangulate(
    _In_ CvPoint2D32f **features2D,
    _In_ int numFeatures,
    _In_ double **cameraCoefficients,
    _In_ int numCameras,
    _Out_ CvPoint3D32f *features3D,
    _InOut_ char *validFeatureIndicator,
    _In_ double errorThreshold)
{    
    // allocate memory and get the u and v coordinates for each camera
    double *u = (double *)malloc(numCameras * sizeof(double));    
    double *v = (double *)malloc(numCameras * sizeof(double));
    
    if (u == NULL || v == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // allocate memory for the R variable used in triangulation calculations
    double *R = (double *)malloc(numCameras * sizeof(double));
    
    if (R == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // we calculate R using the 3D coordinates of the previously calculated feature,
    // so we initiate R to be 1 for the first triangulation calculation
    for (int i = 0; i < numCameras; i++)
    {
        R[i] = 1.0;
    }
    
    // allocate memory for the matrices and vectors we will be using for
    // triangulation
    gsl_matrix *A = gsl_matrix_alloc(numCameras*2, 3);    
    gsl_matrix *QR = gsl_matrix_alloc(numCameras*2, 3); 
    gsl_vector *b = gsl_vector_alloc(numCameras*2);        
    gsl_vector *x = gsl_vector_alloc(3);
    gsl_vector *tau = gsl_vector_alloc(3);
    gsl_vector *residual = gsl_vector_alloc(numCameras*2);
    gsl_vector *bPrime = gsl_vector_alloc(numCameras*2);
    
    if (A == NULL || QR == NULL || b == NULL || x == NULL || tau == NULL || residual == NULL || bPrime == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // set a last valid feature index (as not all 3D coordinates will be valid) to
    // use for the calculation of R
    int lastValidFeatureIndex = -1;    
    
    // loop through each feature
    for (int j = 0; j < numFeatures; j++)
    {
        for (int i = 0; i < numCameras; i++)
        {
            u[i] = features2D[i][j].x;
            v[i] = features2D[i][j].y;
        }
        
        // calculate the value of R using the 3D coordinates of the last valid
        // triangulated feature
        if (lastValidFeatureIndex > -1)
        {
            for (int i = 0; i < numCameras; i++)
            {
                R[i] = cameraCoefficients[i][8]*features3D[lastValidFeatureIndex].x + cameraCoefficients[i][9]*features3D[lastValidFeatureIndex].y + cameraCoefficients[i][10]*features3D[lastValidFeatureIndex].z + 1;
            }
        }
        
        // do triangulation calculations via the DLT method using u,v coordinates 
        // of the feature points, the 11 camera coefficients, and R. Store the
        // results in matrix A and vector b to be linearly solved.
        for (int i = 0; i < numCameras; i++)
        {
            gsl_matrix_set(A, i*2, 0, (u[i]*cameraCoefficients[i][8] - cameraCoefficients[i][0])/R[i]);
            gsl_matrix_set(A, i*2, 1, (u[i]*cameraCoefficients[i][9] - cameraCoefficients[i][1])/R[i]);
            gsl_matrix_set(A, i*2, 2, (u[i]*cameraCoefficients[i][10] - cameraCoefficients[i][2])/R[i]);
            
            gsl_matrix_set(A, i*2+1, 0, (v[i]*cameraCoefficients[i][8] - cameraCoefficients[i][4])/R[i]);
            gsl_matrix_set(A, i*2+1, 1, (v[i]*cameraCoefficients[i][9] - cameraCoefficients[i][5])/R[i]);
            gsl_matrix_set(A, i*2+1, 2, (v[i]*cameraCoefficients[i][10] - cameraCoefficients[i][6])/R[i]);
            
            gsl_vector_set(b, i*2, (cameraCoefficients[i][3] - u[i])/R[i]);
            gsl_vector_set(b, i*2+1, (cameraCoefficients[i][7] - v[i])/R[i]);
        }
        
        // copy matrix A to matrix QR for QR decomposition
        gsl_matrix_memcpy(QR, A);
        
        // perform QR decomposition on A
        gsl_linalg_QR_decomp(QR, tau);
        
        // linearly solve the matrix equation A*x = b for x, which stores the x,y,z
        // coordinates of the feature
        gsl_linalg_QR_lssolve (QR, tau, b, x, residual);
        
        features3D[j].x = gsl_vector_get(x, 0);
        features3D[j].y = gsl_vector_get(x, 1);
        features3D[j].z = gsl_vector_get(x, 2);
        
        // multiply A by x to get b', which contains the ideal centered u,v
        // coordinate of the feature for error calculation
        gsl_blas_dgemv(CblasNoTrans, 1.0, A, x, 0.0, bPrime);
        
        double sum = 0.0;
        
        // calculate the root mean square error between the ideal centered u,v
        // point and the actual u,v point of the feature
        for (int i = 0; i < numCameras*2; i++)
        {
            sum += pow(gsl_vector_get(b, i) - gsl_vector_get(bPrime, i), 2);
        }
        
        int degreesOfFreedom = numCameras*2 - 3;
        double rootMeanSquareError = sqrt(sum / (double)degreesOfFreedom);                
        
        // if the error is greater than the error threshold, set the feature as
        // invalid in the validFeatureIndicator array
        if (rootMeanSquareError > errorThreshold)
        {
            validFeatureIndicator[j] = '\0';
        }
        else
        {
            validFeatureIndicator[j] = '1';
            lastValidFeatureIndex = j;
        }
    }
    
    // cleanup
    free(u);
    free(v);
    
    free(R);
    
    gsl_matrix_free(A);
    gsl_matrix_free(QR);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_vector_free(tau);
    gsl_vector_free(residual);
    gsl_vector_free(bPrime);
    
    return 0;
}

/* Function: readCoefficientsFromInputFile
 * 
 * Description: reads the 11 camera coefficients from n cameras from a file, in \
 *     the format:
 * 
 *     <number of cameras>
 *     camera_1
 *     <1st coefficient of camera 1>
 *     ...
 *     <11th coefficient of camera 1>
 *     ...
 *     camera_n
 *     <1st coefficient of camera n>
 *     ...
 *     <11th coefficient of camera n>
 * 
 * Parameters:
 *     cameraCoefficients: 2D array to store the 11 coefficients read in for each
 *         of n cameras
 *     numCameras: number of cameras, read in through the input file
 *     filename: path of the input file containing the camera coefficients
 * 
 * Returns: 0 on success, error code on error.
 */
int readCoefficientsFromInputFile(
    _Out_ double ***cameraCoefficients,
    _Out_ int *numCameras,
    _In_ char *filename)
{
    // open the file for reading
    FILE *coefficientsFile = fopen(filename, "r");
    
    if (coefficientsFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }
    
    // get the number of cameras
    if (fscanf(coefficientsFile, "%d", &(*numCameras)) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    if (*numCameras < 2)
    {
        return INCORRECT_NUM_CAMERAS_ERROR;
    }
    
    // allocate memory to store the camera coefficients for n cameras
    *cameraCoefficients = (double **)malloc(*numCameras * sizeof(double *));
    
    if (*cameraCoefficients == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    for (int i = 0; i < *numCameras; i++)
    {
        (*cameraCoefficients)[i] = (double *)malloc(NUM_COEFFICIENTS * sizeof(double));
        
        if ((*cameraCoefficients)[i] == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
    }
    
    char stringBuffer[50];
    float coefficientBuffer;
    
    // for each camera, get the 11 coefficients, while making sure the file has
    // correct format
    for (int i = 0; i < *numCameras; i++)
    {
        if (fscanf(coefficientsFile, "%s", stringBuffer) != 1)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        char title[20];
        sprintf(title, "camera_%d", i+1);
        
        if (strcmp(title, stringBuffer) != 0)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        for (int j = 0; j < NUM_COEFFICIENTS; j++)
        {
            if (fscanf(coefficientsFile, "%f", &coefficientBuffer) != 1)
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
            
            (*cameraCoefficients)[i][j] = coefficientBuffer;
        }
    }

    // close the file
    fclose(coefficientsFile);
    
    return 0;
}

/* Function: readFeaturePointsFromInputFile
 * 
 * Description: reads the 2D feature points from n contours in m cameras from an 
 *     input file, as outputted by features.cpp, in the format:
 * 
 *     <number of contours>
 *     camera_1
 *     <contour 1 name> <number of features in contour 1>
 *     <x coordinate of camera 1 ROI 1 feature 1> <y coordinate of camera 1 ROI 1 feature 1>
 *     ...
 *     <x coordinate of camera 1 ROI 1 feature n> <y coordinate of camera 1 ROI 1 feature n>
 *     ...
 *     <contour m name> <number of features in contour m>
 *     <x coordinate of camera 1 ROI m feature 1> <y coordinate of camera 1 ROI m feature 1>
 *     ...
 *     <x coordinate of camera 1 ROI m feature n> <y coordinate of camera 1 ROI m feature n>
 *     camera_2
 *     <contour 1 name> <number of features in contour 1>
 *     <x coordinate of camera 2 ROI 1 feature 1> <y coordinate of camera 2 ROI 1 feature 1>
 *     ...
 *     <x coordinate of camera 2 ROI 1 feature n> <y coordinate of camera 2 ROI 1 feature n>
 *     ...
 *     <contour m name> <number of features in contour m>
 *     <x coordinate of camera 2 ROI m feature 1> <y coordinate of camera 2 ROI m feature 1>
 *     ...
 *     <x coordinate of camera 2 ROI m feature n> <y coordinate of camera 2 ROI m feature n>
 * 
 * Parameters:
 *     features2D: CvPoint3D32f array to store the feature points from n cameras 
 *         read in from the input file for each contour
 *     numFeaturesInContours: number of 2D features in each contour
 *     contourNames: names of the contours
 *     numContours: number of contours
 *     numCameras: number of cameras
 *     filename: path of the input file containing the 2D features
 * 
 * Returns: 0 on success, error code on error.
 */
int readFeaturePointsFromInputFile(
    _Out_ CvPoint2D32f ****features2D,
    _Out_ int **numFeaturesInContours,
    _Out_ char ***contourNames,
    _Out_ int *numContours,
    _In_ int numCameras,
    _In_ char *filename)
{
    // open the feature file for reading
    FILE *featuresFile = fopen(filename, "r");
    
    if (featuresFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }
    
    // get number of contours
    if (fscanf(featuresFile, "%d", &(*numContours)) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    // at least 1 contour required
    if (*numContours < 1)
    {
        return INVALID_NUM_CONTOURS_ERROR;
    }
    
    *features2D = (CvPoint2D32f ***)malloc((*numContours) * sizeof(CvPoint2D32f **));
    *numFeaturesInContours = (int *)malloc((*numContours) * sizeof(int));
    *contourNames = (char **)malloc((*numContours) * sizeof(char *));
    
    if (*features2D == NULL || *numFeaturesInContours == NULL || *contourNames == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // for each contour, get allocate memory to store the features for n cameras
    for (int i = 0; i < *numContours; i++)
    {
        (*features2D)[i] = (CvPoint2D32f **)malloc(numCameras * sizeof(CvPoint2D32f *));        
        (*contourNames)[i] = (char *)malloc(MAX_CONTOUR_NAME_LENGTH * sizeof(char));
        
        if ((*features2D)[i] == NULL || (*contourNames)[i] == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
    }
        
    char title[30];
    char stringBuffer[50];
    float uBuffer, vBuffer;
    
    // for each camera and each contour in each camera, get the 2D feature points
    for (int j = 0; j < numCameras; j++)
    {
        if (fscanf(featuresFile, "%s", stringBuffer) != 1)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        sprintf(title, "camera_%d", j+1);
        
        if (strcmp(title, stringBuffer) != 0)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        for (int i = 0; i < *numContours; i++) 
        {
            // if this is the first camera, get the contour name and number of 
            // features in contour (all cameras should have the same ones in the
            // respective order)
            if (j == 0)
            {
                if (fscanf(featuresFile, "%s %d", (*contourNames)[i], &((*numFeaturesInContours)[i])) != 2)
                {
                    return INCORRECT_INPUT_FILE_FORMAT_ERROR;
                }
            }
            else
            {
                char contourNameBuffer[MAX_CONTOUR_NAME_LENGTH];
                int numFeaturesInContoursBuffer;
                
                if (fscanf(featuresFile, "%s %d", contourNameBuffer, &numFeaturesInContoursBuffer) != 2)
                {
                    return INCORRECT_INPUT_FILE_FORMAT_ERROR;
                }
                
                // if this is not the first camera, check to see if the contour
                // name and number of features in contour is the same as those
                // the first camera
                if (strcmp(contourNameBuffer, (*contourNames)[i]) || (numFeaturesInContoursBuffer != (*numFeaturesInContours)[i]))
                {
                    return INCORRECT_INPUT_FILE_FORMAT_ERROR;
                }
            }
            
            (*features2D)[i][j] = (CvPoint2D32f *)malloc((*numFeaturesInContours)[i] * sizeof(CvPoint2D32f));
            
            if ((*features2D)[i][j] == NULL)
            {
                return OUT_OF_MEMORY_ERROR;
            }
            
            for (int k = 0; k < (*numFeaturesInContours)[i]; k++)
            {
                if (fscanf(featuresFile, "%f %f", &uBuffer, &vBuffer) != 2)
                {
                    return INCORRECT_INPUT_FILE_FORMAT_ERROR;
                }
                
                (*features2D)[i][j][k].x = uBuffer;
                (*features2D)[i][j][k].y = vBuffer;
            }
        }
    }
    
    // close the file
    fclose(featuresFile);
    
    return 0;
}

/* Function: write3DFeaturesToFile
 * 
 * Description: writes the 3D feature points of all contours to a file, in the 
 *     format:
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
 *     features3D: triangulated 3D features for each contour
 *     validFeatureIndicator: Char array where an element has been set to 1 if 
 *         the corresponding feature is valid, 0 if invalid
 *     numFeaturesInContours: number of features in each contour
 *     contourNames: names of the contours
 *     numContours: number of contours
 *     outputFile: path of the output file to write 3D features to
 * 
 * Returns: 0 on success, error code on error.
 */
int write3DFeaturesToFile(
    _In_ CvPoint3D32f **features3D,
    _In_ char **validFeatureIndicator,
    _In_ int *numFeaturesInContours,
    _In_ char **contourNames,
    _In_ int numContours,
    _In_ char *outputFilename)
{
    // open the output file for writing
    FILE *outputFile = fopen(outputFilename, "w");
    
    if (outputFile == NULL)
    {
        return OUTPUT_FILE_OPEN_ERROR;
    }
    
    // write the number of contours
    fprintf(outputFile, "%d\n", numContours);
    
    // for each contour, write the 3D coordinates of the feature points, as well
    // as whether or not they are valid
    for (int i = 0; i < numContours; i++)
    {        
        fprintf(outputFile, "%s %d\n", contourNames[i], numFeaturesInContours[i]);
        
        for (int j = 0; j < numFeaturesInContours[i]; j++)
        {            
            if (validFeatureIndicator[i][j])
            {
                fprintf(outputFile, "1 ");
            }
            else
            {
                fprintf(outputFile, "0 ");
            }
            
            fprintf(outputFile, "%f %f %f\n", features3D[i][j].x, features3D[i][j].y, features3D[i][j].z);
        }
    }
    
    // close output file
    fclose(outputFile);
    
    return 0;
}

/* Function: calculateIdealFeatures
 * 
 * Description: calculates the re-projected ideal 2D features of the triangulated
 *     3D coordinates based on the 11 camera coefficients.
 *
 * Parameters:
 *     idealFeatures2D: array to store the ideal re-projected 2D features
 *     features3D: array of triangulated 3D features
 *     validFeatureIndicator: Char array where an element has been set to 1 if the 
 *         corresponding feature is valid, 0 if invalid. We do not bother finding 
 *         the ideal coordinates of invalid features.
 *     numFeatures: number of features
 *     cameraCoefficients: array of the 11 coefficients of the camera to re-project
 *         the 3D features for.
 */
void calculateIdealFeatures(
    _Out_ CvPoint2D32f *idealFeatures2D,
    _In_ CvPoint3D32f *features3D, 
    _In_ char *validFeatureIndicator,
    _In_ int numFeatures, 
    _In_ double *cameraCoefficients)
{
    // for each valid feature, calculate the ideal u,v coordinates from the
    // triangulated 3D point for a particular camera based on the DLT method, 
    // and store the result in the idealFeatures2D array
    for (int i = 0; i < numFeatures; i++)
    {
        if (validFeatureIndicator[i])
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
}

/* Function: drawCirclesAroundFeatures
 * 
 * Description: Draws over the input image red circles around the actual detected 
 *     features and green circles around the ideal re-projected features after 
 *     triangulation.
 * 
 * Parameters:
 *     image: Input image to circle the actual and ideal features in
 *     actualFeatures2D: CvPoint2D32f array that contains the actual input feature points
 *     idealFeatures2D: CvPoint2D32f array that contains the re-projected ideal feature 
 *         points after triangulation
 *     numFeatures: number of features
 *     validFeatureIndicator: Char array where an element has been set to 1 if the
 *         corresponding feature is valid, 0 otherwise. We do not need to draw
 *         circles around invalid features.
 */
void drawCirclesAroundFeatures(
    _InOut_ IplImage *image, 
    _In_ CvPoint2D32f *actualFeatures2D,
    _In_ CvPoint2D32f *idealFeatures2D, 
    _In_ int numFeatures,
    _In_ char *validFeatureIndicator)
{
    // paramters for drawing circles
    int maxPixelValue;
    
    if (image->depth == IPL_DEPTH_8U)
    {
        maxPixelValue = UCHAR_MAX;
    }
    else
    {
        maxPixelValue = USHRT_MAX;
    }
    
    CvScalar actualFeaturesColor = CV_RGB(maxPixelValue, 0, 0);
    CvScalar idealFeaturesColor = CV_RGB(0, maxPixelValue, 0);
    
    int radius = 1;
    int thickness = 1;
    int lineType = CV_AA;
    
    // for each valid feature, draw red circle around actual feature green circle
    // around ideal feature
    for (int i = 0; i < numFeatures; i++)
    {
        if (validFeatureIndicator[i])
        {
            cvCircle(image, cvPoint((int)actualFeatures2D[i].x, (int)actualFeatures2D[i].y), radius, actualFeaturesColor, thickness, lineType);
            cvCircle(image, cvPoint((int)idealFeatures2D[i].x, (int)idealFeatures2D[i].y), radius, idealFeaturesColor, thickness, lineType);
        }
    }
}

/* Function: drawCamera1ToCamera2FeaturesFlow
 * 
 * Description: Draws the flow vectors between the 2D features of cameras 1 and 2. 
 *     Used for debugging purposes.
 * 
 * Parameters:
 *     image: Input image to draw the flow vectos on
 *     features2D: Array of 2D features for cameras 1 and 2
 *     numFeatures: number of features
 *     validFeatureIndicator: Char array where an element has been set to 1 if the
 *         corresponding feature is valid, 0 if invalid. We do not need to draw
 *         circles around invalid features.
 */
void drawCamera1ToCamera2FeaturesFlow(
    _InOut_ IplImage *image, 
    _In_ CvPoint2D32f **features2D,
    _In_ int numFeatures,
    _In_ char *validFeatureIndicator)
{
    // loop through each feature
    for(int i = 0; i < numFeatures; i++)
    {
        // only draw flow line if the features is valid
        if (validFeatureIndicator[i])
        {
            // 'p' is the point of the actual feature. 'q' is the point of the
            // corresponding ideal feature.
            CvPoint p, q;
            p.x = (int) features2D[0][i].x;
            p.y = (int) features2D[0][i].y;
            q.x = (int) features2D[1][i].x;
            q.y = (int) features2D[1][i].y;

            // only draw flow lines that are larger than the specified minimum 
            // flow length
            int minFlowLength = 0;
            
            // the flow lines may be too short for a good visualization due to 
            // high framerate between the two frames, so we can lengthen them by 
            // a lengthening factor
            int lengtheningFactor = 1;
             
            // scaling for the arrown tips so that they proportional to the flow 
            // lines
            int arrowScale = 5;
            
            // draw the flow line with arrows
            drawFlowArrows(image, p, q, minFlowLength, lengtheningFactor, arrowScale);
        }
    }
}

/* Function: displayImageWithActualAndIdealFeatures
 * 
 * Description: Displays the image from one camera, with the actual and ideal 
 *     features highlighted on it with different colored circles
 * 
 * Parameters:
 *     image: Input image to display and show features on.
 *     numContours: number of contours
 *     actualFeatures2D: CvPoint2D32f array that contains the actual input feature 
 *         points
 *     idealFeatures2D: CvPoint2D32f array that contains the re-projected ideal 
 *         feature points after triangulation
 *     numFeatures: number of features for each contour
 *     validFeatureIndicator: Char array where an element has been set to 1 if the
 *         corresponding feature is valid, 0 if invalid.
 */
void displayImageWithActualAndIdealFeatures(
    _In_ IplImage *image,
    _In_ int numContours,
    _In_ CvPoint2D32f ***actualFeatures2D,
    _In_ CvPoint2D32f **idealFeatures2D, 
    _In_ int *numFeaturesInContours,
    _In_ char **validFeatureIndicator)
{
    // create a display image by cloning the original image
    IplImage *displayImage = cvCloneImage(image);    
    char displayWindowName[] = "Displaying valid features in red and their reprojections in green";
    
    cvNamedWindow(displayWindowName);
    cvMoveWindow(displayWindowName, 0,0);
    
    for (int i = 0; i < numContours; i++)
    {
        drawCirclesAroundFeatures(displayImage, actualFeatures2D[i][0], idealFeatures2D[i], numFeaturesInContours[i], validFeatureIndicator[i]);
    }    
    
    // flip image over x axis to display in original coordinates
    cvFlip(displayImage, NULL, 0);
    
    // display the image in a window
    cvShowImage(displayWindowName, displayImage);
    cvWaitKey(0);
    
    // destroy the window and free the display image
    cvDestroyWindow(displayWindowName);
    cvReleaseImage(&displayImage);
}

/* Function: isOpticFlowValid
 * 
 * Description: Check if the specified optic flow property (either length or angle) 
 *     between features in a set of 2 camera images is between specified min and max 
 *     values.
 * 
 * Parameters:
 *     c1Features: array of features for camera 1
 *     c2Features: array of features for camera 2
 *     numFeatures: number of features
 *     validFeatureIndicator: char array indicating whether a feature is valid
 *         or not. Value for corresponding feature is set to 0 if the optic flow 
 *         property for a feature is smaller than the minimum value or larger 
 *         than the maximum value.
 *     minAcceptedValue: minimum accepted flow property value.
 *     maxAcceptedValue: maximum accepted flow property value.
 *     opticFlowProperty: specifies which optic flow property (length or angle)
 *         to check the validity of.
 */
void isOpticFlowValid(
    _In_ CvPoint2D32f *c1Features,
    _In_ CvPoint2D32f *c2Features,
    _In_ int numFeatures,
    _InOut_ char *validFeatureIndicator,
    _In_ double minAcceptedValue,
    _In_ double maxAcceptedValue,
    _In_ int opticFlowProperty)
{
    // loop through each feature
    for(int i = 0; i < numFeatures; i++)
    {
        // don't bother checking for accepted flow property if feature is already
        // found to be invalid
        if (validFeatureIndicator[i])
        {
            CvPoint2D32f c1CurPoint = c1Features[i];
            CvPoint2D32f c2CurPoint = c2Features[i];

            double opticFlowPropertyValue;
            
            if (opticFlowProperty == FLOW_LENGTH)
            {
                // get the length of the optic flow
                opticFlowPropertyValue = sqrt(pow(c1CurPoint.y - c2CurPoint.y, 2) + pow(c1CurPoint.x - c2CurPoint.x, 2));   
            }
            else
            {
                // get the angle of the optic flow
                opticFlowPropertyValue = atan2((double)(c1CurPoint.y - c2CurPoint.y), (double)(c1CurPoint.x - c2CurPoint.x));
            }
                
            // if the property is less than the minimum accepted value or greater
            // than the maximum accepted value, set corresponding value for feature 
            // in 'validFeatureIndicator' to 0.
            if (opticFlowPropertyValue < minAcceptedValue || opticFlowPropertyValue > maxAcceptedValue)
            {
                validFeatureIndicator[i] = 0;
            }

        }
    }    
}

/* Function: doublesComparator
 * 
 * Description: Simple comparator for doubles to be passed to qsort function. 
 * 
 * Parameters:
 *     a: first value to compare
 *     b: second value to compare
 * 
 * Returns: A negative value if a < b, 0 if a == b, or a positive value if a > b.
 */
int doublesComparator (
    _In_ const void *a, 
    _In_ const void *b)
{
    return (*(double *)a - *(double *)b);
}

/* Function: getFlowMedianAndAverageDeviation
 * 
 * Description: Get the median specified optic flow property from a set of features,
 *     as well as the absolute average deviation (calculated from the median).
 * 
 * Parameters:
 *     c1Features: array of features for camera 1
 *     c2Features: array of features for camera 2
 *     numFeatures: number of features
 *     validFeatureIndicator: char array indicating whether a feature is valid
 *         or not.
 *     median: median of the flow property values, set by this function.
 *     averageDeviation: average deviation from the median of the flow property
 *         values, set by this function.
 *     opticFlowProperty: specifies which optic flow property (length or angle)
 *         to get the median and average deviation of.
 */
void getFlowMedianAndAverageDeviation(
    _In_ CvPoint2D32f *c1Features,
    _In_ CvPoint2D32f *c2Features,
    _In_ int numFeatures,
    _InOut_ char *validFeatureIndicator,
    _Out_ double *median,
    _Out_ double *averageDeviation,
    _In_ int opticFlowProperty)
{
    // array to store all optic flow property values of valid features
    double opticFlowPropertyValues[numFeatures];
    int numValidFeatures = 0;
       
    // loop through each feature
    for(int i = 0; i < numFeatures; i++)
    {
        // only process features that are valid
        if (validFeatureIndicator[i])
        {
            CvPoint2D32f c1CurPoint = c1Features[i];
            CvPoint2D32f c2CurPoint = c2Features[i];

            if (opticFlowProperty == FLOW_LENGTH)
            {
                // get the length of the optic flow
                double opticFlowLength = sqrt(pow(c1CurPoint.y - c2CurPoint.y, 2) + pow(c1CurPoint.x - c2CurPoint.x, 2));

                opticFlowPropertyValues[numValidFeatures] = opticFlowLength;
                numValidFeatures++;
            }
            else
            {
                // get the angle of the optic flow
                double opticFlowAngle = atan2((double)(c1CurPoint.y - c2CurPoint.y), (double)(c1CurPoint.x - c2CurPoint.x));
                
                opticFlowPropertyValues[numValidFeatures] = opticFlowAngle;
                numValidFeatures++;
            }
        }
    }
    
    // sort the values of the specified optic flow property array to find the 
    // median
    qsort(opticFlowPropertyValues, numValidFeatures, sizeof(double), doublesComparator);
    
    if (numValidFeatures == 0)
    {
        *median = 0;
    }
    else
    {
        *median = opticFlowPropertyValues[(int)(numValidFeatures/2)];
    }
    
    double sum = 0;
    
    // calculate the absolute average deviation from the median
    for (int i = 0; i< numValidFeatures; i++)
    {
        sum += fabs(opticFlowPropertyValues[i] - *median);
    }
    
    *averageDeviation = sum / (double) numValidFeatures;
}

/* Function: getFlowMeanAndStandardDeviation
 * 
 * Description: Get the mean specified optic flow property from a set of features,
 *     as well as the standard deviation.
 * 
 * Parameters:
 *     c1Features: array of features for camera 1
 *     c2Features: array of features for camera 2
 *     numFeatures: number of features
 *     validFeatureIndicator: char array indicating whether a feature is valid
 *         or not.
 *     mean: mean of the flow property values, set by this function.
 *     standardDeviation: standard deviation from the median of the flow property
 *         values, set by this function.
 *     opticFlowProperty: specifies which optic flow property (length or angle)
 *         to get the mean and standard deviation of.
 */
void getFlowMeanAndStandardDeviation(
    _In_ CvPoint2D32f *c1Features,
    _In_ CvPoint2D32f *c2Features,
    _In_ int numFeatures,
    _InOut_ char *validFeatureIndicator,
    _Out_ double *mean,
    _Out_ double *standardDeviation,
    _In_ int opticFlowProperty)
{
    // array to store all optic flow property values of valid features
    double opticFlowPropertyValues[numFeatures];
    int numValidFeatures = 0;

    double sum = 0;
       
    // loop through each feature
    for(int i = 0; i < numFeatures; i++)
    {
        // only process features that are valid
        if (validFeatureIndicator[i])
        {
            CvPoint2D32f c1CurPoint = c1Features[i];
            CvPoint2D32f c2CurPoint = c2Features[i];
            
            if (opticFlowProperty == FLOW_LENGTH)
            {
                // get the length of the optic flow
                double opticFlowLength = sqrt(pow(c1CurPoint.y - c2CurPoint.y, 2) + pow(c1CurPoint.x - c2CurPoint.x, 2));
                sum += opticFlowLength;
                    
                opticFlowPropertyValues[numValidFeatures] = opticFlowLength;
                numValidFeatures++;

            }
            else
            {
                // get the angle of the optic flow
                double opticFlowAngle = atan2((double)(c1CurPoint.y - c2CurPoint.y), (double)(c1CurPoint.x - c2CurPoint.x));                
                sum += opticFlowAngle;
                
                opticFlowPropertyValues[numValidFeatures] = opticFlowAngle;
                numValidFeatures++;
            }
        }
    }
    
    // calculate the mean of the optic flow property of all valid features
    *mean = sum / (double) numValidFeatures;
    
    sum = 0;
    
    // calculate the standard deviation
    for(int i = 0; i < numValidFeatures; i++)
    {
        sum += pow(opticFlowPropertyValues[i] - *mean, 2);
    }

    *standardDeviation = sqrt(sum / (double) numValidFeatures);
}

/* Function: getNumValidFeatures
 * 
 * Description: count the number of set chars in validFeatureIndicator to determine
 *     how many features are valid
 * 
 * Parameters:
 *     numFeatures: number of total (valid and invalid) features
 *     validFeatureIndicator: char array to indicate if feature is valid or not
 * 
 * Returns: number of valid features
 */
int getNumValidFeatures(
    _In_ int numFeatures,
    _In_ char *validFeatureIndicator)
{
    int numValidFeatures = 0;
    
    for (int i = 0; i < numFeatures; i++)
    {
        if (validFeatureIndicator[i])
        {
            numValidFeatures++;
        }
    }
    
    return numValidFeatures;
}

/* Function: triangulation
 * 
 * Description: read in 2D points from n cameras and camera coefficients from file.
 *     Triangulate the 2D points into 3D points. We determine invalid points through
 *     triangulation residual checking and outlier checking. We write the triangulated 
 *     3D features to file, along with an indicator on whether it is valid or not.
 * 
 * Parameters: 
 *     cameraCoefficientsFilename: filename of the camera coefficients
 *     featuresFilename: filename of the 2D features
 *     feature3DFilename: filename of the output file to write the 3D features to
 *     displayImage: image to display resulting triangulated points on for
 *         debugging purposes.
 *     errorMessage: string to output an error message to, on error
 * 
 * Returns: 0 on success, 1 on error.
 */
int triangulation(
    _In_ char *cameraCoefficientsFilename, 
    _In_ char *featuresFilename, 
    _In_ char *features3DFilename,
    _In_ IplImage *displayImage,
    _Out_ char *errorMessage)
{    
    // variable to store error status returned from functions
    int status;
    
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
    
    int numContours;
    CvPoint2D32f ***features2D;
    int *numFeaturesInContours;
    char **contourNames;
    
    // get the number of features and feature points for each camera for each
    // contour from file
    status = readFeaturePointsFromInputFile(&features2D, &numFeaturesInContours, &contourNames, &numContours, numCameras, featuresFilename);
    
    if (status == INPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open feature points file.");
        return 1;
    }    
    
    if (status == INVALID_NUM_CONTOURS_ERROR)
    {
        sprintf(errorMessage, "At least 1 contour region is required.");
        return 1;
    }
    
    if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
    {
        sprintf(errorMessage, "Feature points file has incorrect format.");
        return 1;
    }    
    
    if (status == OUT_OF_MEMORY_ERROR)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    CvPoint3D32f **features3D = (CvPoint3D32f **)malloc(numContours * sizeof(CvPoint3D32f *));
    
    if (features3D == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    char **validFeatureIndicator = (char **)malloc(numContours * sizeof(char *));
    
    if (validFeatureIndicator == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    // calculate the error threshold for triangulation residual based on the 
    // resolution of the original images
    double errorThreshold = (displayImage->width * displayImage->height) / FACTOR;
    
    // for each contour, perform triangulation and invalid feature removal
    for (int i = 0; i < numContours; i++)
    {
        features3D[i] = (CvPoint3D32f *)malloc(numFeaturesInContours[i] * sizeof(CvPoint3D32f));        
        validFeatureIndicator[i] = (char *)malloc(numFeaturesInContours[i] * sizeof(char));
        
        if (features3D[i] == NULL || validFeatureIndicator[i] == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        // triangulate for the 3D features using the input 2D coordinates
        status = triangulate(features2D[i], numFeaturesInContours[i], cameraCoefficients, numCameras, features3D[i], validFeatureIndicator[i], errorThreshold);
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        int numValidFeatures = getNumValidFeatures(numFeaturesInContours[i], validFeatureIndicator[i]);
        printf("Valid features after triangulation for contour %s: %d\n", contourNames[i], numValidFeatures);
        
        
        // display the flow of valid 2D features (after triangulation) between 
        // cameras 1 and 2, used for debugging purposes
        /*IplImage *cameraFlowImageBeforeOutlierRemoval = cvCloneImage(displayImage);
        drawCamera1ToCamera2FeaturesFlow(cameraFlowImageBeforeOutlierRemoval, features2D[i], numFeaturesInContours[i], validFeatureIndicator[i]);
        
        cvFlip(cameraFlowImageBeforeOutlierRemoval, NULL, 0);
        cvShowImage("image", cameraFlowImageBeforeOutlierRemoval);   
        cvWaitKey(0);
    
        cvDestroyWindow("image");    
        cvReleaseImage(&cameraFlowImageBeforeOutlierRemoval);*/
        
        // find outliers (using standard deviation or average deviation in flow 
        // length and/or flow angle) in the flows of 2D features between cameras 
        // 1 and 2, set outliers as invalid
        for (int j = 0; j < numCameras; j++)
        {
            for (int k = j + 1; k < numCameras; k++)
            {
                // Since features should have moved roughly the same distance in roughtly the
                // same direction between cameras 1 and 2, we do some statistical calculations
                // on the optic flow lengths and angles, and we filter out the features that 
                // moved in too different of a direction or distance from the rest of the 
                // features. Those features are likely to not be consistent between the frames 
                // of cameras 1 and 2.    
                double lengthMean;
                double lengthStandardDeviation;
                
                getFlowMeanAndStandardDeviation(features2D[i][j], features2D[i][k], numFeaturesInContours[i], validFeatureIndicator[i], &lengthMean, &lengthStandardDeviation, FLOW_LENGTH);
                
                double minAcceptedFlowLength = lengthMean - lengthStandardDeviation;
                double maxAcceptedFlowLength = lengthMean + lengthStandardDeviation;
                
                isOpticFlowValid(features2D[i][j], features2D[i][k], numFeaturesInContours[i], validFeatureIndicator[i], minAcceptedFlowLength, maxAcceptedFlowLength, FLOW_LENGTH);
            }
        }
                
        numValidFeatures = getNumValidFeatures(numFeaturesInContours[i], validFeatureIndicator[i]);
        printf("Valid features after removing flow outliers for contour %s: %d\n", contourNames[i], numValidFeatures);
        
        
        // display the flow of valid 2D features (after outlier removal) between 
        // cameras 1 and 2, used for debugging purposes
        /*IplImage *cameraFlowImageAfterOutlierRemoval = cvCloneImage(displayImage);
        drawCamera1ToCamera2FeaturesFlow(cameraFlowImageAfterOutlierRemoval, features2D[i], numFeaturesInContours[i], validFeatureIndicator[i]);
        
        cvFlip(cameraFlowImageAfterOutlierRemoval, NULL, 0);
        cvShowImage("image", cameraFlowImageAfterOutlierRemoval);   
        cvWaitKey(0);
    
        cvDestroyWindow("image");    
        cvReleaseImage(&cameraFlowImageAfterOutlierRemoval);*/
    }
    
    // display the camera 1 image with actual 2D features and reprojected 2D
    // features calculated after triangulation
    if (displayImage->depth != IPL_DEPTH_8U && displayImage->depth != IPL_DEPTH_16U)
    {
        printf("Display image depth not supported.\n");
    }
    
    else
    {
        CvPoint2D32f **idealFeatures2D = (CvPoint2D32f **)malloc(numContours * sizeof(CvPoint2D32f *));
        
        if (idealFeatures2D == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        // calculate the ideal 2D features for the 3D features of each contour
        for (int i = 0; i < numContours; i++)
        {
            idealFeatures2D[i] = (CvPoint2D32f *)malloc(numFeaturesInContours[i] * sizeof(CvPoint2D32f));
            
            if (idealFeatures2D[i] == NULL)
            {
                sprintf(errorMessage, "Out of memory error.");
                return 1;
            }
            
            calculateIdealFeatures(idealFeatures2D[i], features3D[i], validFeatureIndicator[i], numFeaturesInContours[i], cameraCoefficients[0]);
        }
        
        // display an RGB image with different colored circles representing 
        // re-projected features and actual features
        IplImage *displayImageRGB = cvCreateImage(cvSize(displayImage->width, displayImage->height), displayImage->depth, 3);
        cvCvtColor(displayImage, displayImageRGB, CV_GRAY2BGR);
        
        displayImageWithActualAndIdealFeatures(displayImageRGB, numContours, features2D, idealFeatures2D, numFeaturesInContours, validFeatureIndicator);                    
        
        cvReleaseImage(&displayImageRGB);
        
        for (int i = 0; i < numContours; i++)
        {
            free(idealFeatures2D[i]);
        }
        
        free(idealFeatures2D);
    }
    
    // write the triangulated 3D features to file
    status = write3DFeaturesToFile(features3D, validFeatureIndicator, numFeaturesInContours, contourNames, numContours, features3DFilename);
    
    if (status == OUTPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open output file.");
        return 1;
    }
    
    // cleanup
    for (int i = 0; i < numContours; i++)
    {
        for (int j = 0; j < numCameras; j++)
        {
            free(features2D[i][j]);
        }
        
        free(features2D[i]);        
        free(features3D[i]);
        free(validFeatureIndicator[i]);
        free(contourNames[i]);
    }
    
    for (int j = 0; j < numCameras; j++)
    {   
        free(cameraCoefficients[j]);
    }
    
    free(features2D);
    free(features3D);
    free(validFeatureIndicator);
    free(cameraCoefficients);    
    free(numFeaturesInContours);
    free(contourNames);
    
    return 0;
} 

