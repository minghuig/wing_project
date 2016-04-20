/* File: ransac.cpp
 *
 * Description: This file contains the program that takes a 3D point cloud, along
 *     with a char array indicating whether each point is valid. It performs the
 *     RANSAC algorithm on the point cloud, trying to fit the points to a model
 *     plane. Upon finding the best-fit model plane, it will set all points that
 *     lie over a certain distance from the plane as invalid.
 *
 * Author: Ming Guo
 * Created: 1/11/12
 */

#include "videos.h"
#include "ransac.h"

#define MIN_VALID_POINTS 40
#define DISTANCE_THRESHOLD_SMALL_OBSERVATION_SET 3
#define DISTANCE_THRESHOLD_LARGE_OBSERVATION_SET 5
#define NUM_ITERATIONS 1000
#define MIN_POINTS_NEEDED_FOR_MODEL 3
#define NUM_LINEAR_EQUATION_PARAMS_FOR_MODEL 4

/* Function: isMember
 * 
 * Description: Check whether point is a member of a list of points
 * 
 * Parameters:
 *     p: point to check if it's a member of the list
 *     pList: list of points
 *     listSize: size of the point list
 * 
 * Returns: true if p is a member of pList, false otherwise
 */
bool isMember(
    _In_ CvPoint3D32f p, 
    _In_ CvPoint3D32f *pList, 
    _In_ int listSize)
{
    bool isMember = false;

    for (int i = 0; i < listSize; i++) 
    {
        if (p.x == pList[i].x && p.y == pList[i].y && p.z == pList[i].z)
        {
            isMember = true;
            break;
        }
    }
    
    return isMember;
}

/* Function: crossProduct
 * 
 * Description: calculates the cross product of 2 vectors of length 3
 * 
 * Parameters:
 *     result: the vector containing the result of the cross product calculation
 *     a: first vector in the cross product calculation
 *     b: second vector in the cross product calculation
 */
void crossProduct(
    _Out_ gsl_vector *result, 
    _In_ gsl_vector *a, 
    _In_ gsl_vector *b)
{    
    // standard computation of cross product
    double ai, aj, ak;
    double bi, bj, bk;
    
    ai = gsl_vector_get(a, 0);
    aj = gsl_vector_get(a, 1);
    ak = gsl_vector_get(a, 2);
    
    bi = gsl_vector_get(b, 0);
    bj = gsl_vector_get(b, 1);
    bk = gsl_vector_get(b, 2);    
    
    gsl_vector_set(result, 0, aj*bk-ak*bj);
    gsl_vector_set(result, 1, ak*bi-ai*bk);
    gsl_vector_set(result, 2, ai*bj-aj*bi);
}

/* Function: getModel
 * 
 * Description: Get linear equation parameters for a plane either directly from 
 *     3 points (which would all be co-planar) or from a least squares fit on >3 
 *     points
 * 
 * Parameters:
 *     modelPlane: the resulting parameters of the model plane calculated from 
 *         the points, as a vector of length 4. We define a plane as 
 *         Ax + By + Cz + D = 0, and the vector contains the values of A, B, C, 
 *         and D
 *     maybeInliers: the points to calculate the model plane from
 *     numPoints: number of points to calculate the model plane from
 * 
 * Returns: 0 on success, error code on error
 */
int getModel(
    _Out_ gsl_vector *modelPlane, 
    _In_ CvPoint3D32f *maybeInliers, 
    _In_ int numPoints)
{
    // need at least 3 points to find a model plane
    if (numPoints < 3)
    {
        return NOT_ENOUGH_POINTS_ERROR;
    }
    
    // if number of points equals 3, find the plane through the given points. 
    if (numPoints == 3)
    {        
        gsl_vector *AB = gsl_vector_alloc(3);
        gsl_vector *AC = gsl_vector_alloc(3);        
        gsl_vector *cp = gsl_vector_alloc(3);
        
        if (AB == NULL || AC == NULL || cp == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
        
        // We use n = (p2 - p1) X (p3 - p1) to find the normal of the plane
        gsl_vector_set(AB, 0, maybeInliers[1].x - maybeInliers[0].x);
        gsl_vector_set(AB, 1, maybeInliers[1].y - maybeInliers[0].y);
        gsl_vector_set(AB, 2, maybeInliers[1].z - maybeInliers[0].z);
        
        gsl_vector_set(AC, 0, maybeInliers[2].x - maybeInliers[0].x);
        gsl_vector_set(AC, 1, maybeInliers[2].y - maybeInliers[0].y);
        gsl_vector_set(AC, 2, maybeInliers[2].z - maybeInliers[0].z);        
        
        crossProduct(cp, AB, AC);
        
        // plug in the coordinate values of one of the points into the normal
        // equation for the plane to find the value of D
        double d = -1 * gsl_vector_get(cp, 0) * maybeInliers[0].x - gsl_vector_get(cp, 1) * maybeInliers[0].y - gsl_vector_get(cp, 2) * maybeInliers[0].z;
        
        // set A, B, C, and D into the result vector
        gsl_vector_set(modelPlane, 0, gsl_vector_get(cp, 0));
        gsl_vector_set(modelPlane, 1, gsl_vector_get(cp, 1));
        gsl_vector_set(modelPlane, 2, gsl_vector_get(cp, 2));
        gsl_vector_set(modelPlane, 3, d);
        
        gsl_vector_free(AB);
        gsl_vector_free(AC);
        gsl_vector_free(cp);
    }
    
    // if number of points is greater than 3, we perform a least squares fit on
    // the points to find the model plane linear equation parameters
    else
    {   
        gsl_matrix *X = gsl_matrix_alloc(numPoints, 3);
        gsl_vector *y = gsl_vector_alloc(numPoints);

        gsl_vector *c = gsl_vector_alloc(3);
        gsl_matrix *cov = gsl_matrix_alloc(3, 3);
        
        if (X == NULL || y == NULL || c == NULL || cov == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
        
        for (int i = 0; i < numPoints; i++)
        {
            gsl_matrix_set(X, i, 0, maybeInliers[i].x);
            gsl_matrix_set(X, i, 1, maybeInliers[i].y);            
            gsl_matrix_set(X, i, 2, maybeInliers[i].z);
            
            gsl_vector_set(y, i, 1.0);
        }
        
        double chisq;
        gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(numPoints, 3);
        
        if (work == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
        
        // perform the multi-parameter fitting
        gsl_multifit_linear (X, y, c, cov, &chisq, work);
        
        // set A, B, C, and D from the fit into the result vector
        gsl_vector_set(modelPlane, 0, gsl_vector_get(c, 0));
        gsl_vector_set(modelPlane, 1, gsl_vector_get(c, 1));
        gsl_vector_set(modelPlane, 2, gsl_vector_get(c, 2));
        gsl_vector_set(modelPlane, 3, -1.0);
        
        gsl_matrix_free(X);
        gsl_matrix_free(cov);
        gsl_vector_free(y);
        gsl_vector_free(c);        
        gsl_multifit_linear_free(work);
    }
    
    return 0;
}

/* Function: timeSeed
 * 
 * Description: Compute a seed for srand based on current system time. If we don't
 *     provide a seed to srand, srand will give us the same random values every 
 *     time the program is ran.
 * 
 * Returns: seed for srand computed from the current system time
 */
unsigned int timeSeed()
{
    time_t now = time (0);
    unsigned char *p = (unsigned char *)&now;
    unsigned int seed = 0;
    size_t i;
 
    for (i = 0; i < sizeof(now); i++)
    {
        seed = seed * (UCHAR_MAX + 2U) + p[i];
    }
 
    return seed;
}

/* Function: getMaybeInliers
 * 
 * Description: Get 3 random points from observation set to build a model plane
 *     from
 * 
 * Parameters:
 *     maybeInliers: The resulting set of 3 randomly selected points from the
 *         observation set
 *     pointCloud: the observation set
 *     validFeatureIndicator: char array indicating which points in the observation
 *         set are valid, do not select points that are invalid
 *     numPoints: number of points in observation set
 * 
 * Returns: 0 on success, error code on error
 */
int getMaybeInliers(
    _Out_ CvPoint3D32f *maybeInliers, 
    _In_ int numMaybeInliers,
    _In_ CvPoint3D32f *pointCloud, 
    _In_ char *validFeatureIndicator, 
    _In_ int numPoints)
{
    int *randomIndices = (int *)malloc(numMaybeInliers * sizeof(int));
    
    if (randomIndices == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // get random points from observation set by selecting random indices, while 
    // making sure that the point is valid and that no repeated points are selected
    for (int i = 0; i < numMaybeInliers; i++)
    {
        int randomIndex;
        bool used;
        
        do
        {
            randomIndex = rand() % numPoints;        
            used = false;
            
            for (int j = 0; j < i; j++)
            {
                if (randomIndices[j] == randomIndex)
                {
                    used = true;
                    break;
                }
            }
        }
        while (used || !validFeatureIndicator[randomIndex]);
        
        randomIndices[i] = randomIndex;
        maybeInliers[i] = pointCloud[randomIndex];
    }
    
    free(randomIndices);
    
    return 0;
}

/* Function: getDistance
 * 
 * Description: Get distance between point and plane
 * 
 * Parameters:
 *     p: point
 *     modelPlane: plane
 * 
 * Returns: distance between point and plane
 */
double getDistance(
    _In_ CvPoint3D32f p, 
    _In_ gsl_vector *modelPlane)
{   
    double a, b, c, d;
    
    a = gsl_vector_get(modelPlane, 0);
    b = gsl_vector_get(modelPlane, 1);
    c = gsl_vector_get(modelPlane, 2);
    d = gsl_vector_get(modelPlane, 3);
    
    // standard distance formula
    return fabs(a * p.x + b * p.y + c * p.z + d) / sqrt(a*a + b*b + c*c);
}

/* Function: fitsModel
 * 
 * Description: Check whether distance of point from model plane is smaller than 
 *     a given threshold
 * 
 * Parameters:
 *     p: point
 *     modelPlane: plane
 * 
 * Returns: true if the point lies within the distance threshold of the model plane
 *     false otherwise
 */
bool fitsModel(
    _In_ CvPoint3D32f p, 
    _In_ gsl_vector *modelPlane,
    _In_ int distanceThreshold)
{
    bool fits = false;
    double distance = getDistance(p, modelPlane);

    if (distance < distanceThreshold) 
    {
        fits = true;
    }

    return fits;
}

/* Function: getModelError
 * 
 * Description: get the model error, defined as the number of inlier points that
 *     fits the model subtracted from a constant
 * 
 * Parameters:
 *     pointList: list of inlier points for the model plane
 *     numPoints: number of points in pointList
 *     modelPlane: linear equation parameters for the model plane
 * 
 * Returns: the model error value
 */
int getModelError(
    _In_ CvPoint3D32f *pointList, 
    _In_ int numPoints, 
    _In_ gsl_vector *modelPlane,
    _In_ int distanceThreshold)
{
    int maxError = INT_MAX;
    int numFits = 0;
    
    // checks to see if each inlier point fits the model plane
    for (int i = 0; i < numPoints; i++) 
    {
        if (fitsModel(pointList[i], modelPlane, distanceThreshold)) 
        {
            numFits++;
        }
    }
    
    return maxError - numFits;
}

/* Function: ransac
 * 
 * Description: RANSAC algorithm (implemented from the pseudocode at 
 *     http://en.wikipedia.org/wiki/RANSAC), using a plane as model (the shape of
 *     the model may be changed by replacing the getModel and fitsModel functions), 
 *     applied to a cloud of points. A best fit model plane is determined through 
 *     N iterations of the algorithm, and points that lie over a certain distance 
 *     threshold from that model are set as invalid.
 * 
 * Parameters:
 *     pointCloud: CvPoint3D32f array of observation points
 *     validFeatureIndicator: char array indication whether each observation point
 *         is valid or not
 *     numPoints: number of observation points
 *     errorMessage: string to output an error message to, on error
 * 
 * Returns: 0 on success, 1 on error
 */
int ransac(
    _In_ CvPoint3D32f *pointCloud, 
    _InOut_ char *validFeatureIndicator, 
    _In_ int numPoints,
    _Out_ char *errorMessage)
{       
    int numValidPoints = 0;
    
    // find the number of valid points in the observation set
    for (int i = 0; i < numPoints; i++)
    {
        if (validFeatureIndicator[i])
        {
            numValidPoints++;
        }
    }
    
    // if number of valid points is less than a set constant, then we return an
    // error because RANSAC will have a significant chance of giving us invalid
    // results
    if (numValidPoints < MIN_VALID_POINTS)
    {
        sprintf(errorMessage, "Not enough valid features to fit model.\n");
        return 1;
    }    
    
    // set the number of required inliers for a randomly selected model plane in 
    // order for it to be accepted as a potential best fit model to be the number
    // of valid observation points divided by 3        
    int numRequiredInliers = numValidPoints/3;
    
    // if the observation set is large enough, we set the distance threshold within
    // which we will accept points off the model plane to be higher to allow for
    // more curvature of the surface. We set the threshold to be lower for a smaller
    // observation set because its error threshold is lower. Adjust values in
    // file definitions as fit.
    int distanceThreshold;
    
    if (numValidPoints < MIN_VALID_POINTS*3)
    {
        distanceThreshold = DISTANCE_THRESHOLD_SMALL_OBSERVATION_SET;
    }
    else
    {
        distanceThreshold = DISTANCE_THRESHOLD_LARGE_OBSERVATION_SET;
    }

    int bestError = INT_MAX;
    gsl_vector *bestModel = gsl_vector_alloc(NUM_LINEAR_EQUATION_PARAMS_FOR_MODEL);
    
    if (bestModel == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    int numMaybeInliers = MIN_POINTS_NEEDED_FOR_MODEL;
    CvPoint3D32f *maybeInliers = (CvPoint3D32f *)malloc(numMaybeInliers * sizeof(CvPoint3D32f));
    
    if (maybeInliers == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
            
    int consensusSetSize;    
    CvPoint3D32f *consensusSet = (CvPoint3D32f *)malloc(numValidPoints * sizeof(CvPoint3D32f));
    
    if (consensusSet == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
        
    gsl_vector *model = gsl_vector_alloc(NUM_LINEAR_EQUATION_PARAMS_FOR_MODEL);
    
    if (model == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    int foundModel = false;
    int iterations = 0;
        
    // initialize the pseudo-random number generator using a seed based on current
    // system time
    srand(timeSeed());
    
    // iterate N times to find a best fit model plane for the observation set
    while (iterations < NUM_ITERATIONS)
    {
        // randomly select 3 points from the observation set
        if (getMaybeInliers(maybeInliers, numMaybeInliers, pointCloud, validFeatureIndicator, numPoints))
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        for (int i = 0; i < numMaybeInliers; i++)
        {
            consensusSet[i] = maybeInliers[i];
        }
        
        consensusSetSize = numMaybeInliers;
        
        // get model plane parameters fitted to the random maybe-inliers
        int status = getModel(model, maybeInliers, numMaybeInliers);
        
        if (status == NOT_ENOUGH_POINTS_ERROR)
        {
            sprintf(errorMessage, "At least %d points are required to define a model.", MIN_POINTS_NEEDED_FOR_MODEL);
            return 1;
        }
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }

        // for each valid point in observation set, if point fits maybe-model 
        // with an error smaller than the threshold, add point to consensus set
        for (int i = 0; i < numPoints; i++) 
        {
            if (validFeatureIndicator[i])
            {
                if (!isMember(pointCloud[i], maybeInliers, numMaybeInliers)) 
                {
                    if (fitsModel(pointCloud[i], model, distanceThreshold)) 
                    {
                        consensusSet[consensusSetSize] = pointCloud[i];
                        consensusSetSize++;
                    }
                }
            }
        }

        // if the consensus set is greater than or equal to the pre-determined
        // number of points needed to accept a model, it implies that we may have 
        // found a good model and we now need to test how good the model is
        if (consensusSetSize >= numRequiredInliers) 
        {
            // re-fit the model plane to all the points in the consensus set
            int status = getModel(model, consensusSet, consensusSetSize);
            
            if (status == NOT_ENOUGH_POINTS_ERROR)
            {
                sprintf(errorMessage, "At least 3 points are required to define a model plane.");
                return 1;
            }
            
            if (status == OUT_OF_MEMORY_ERROR)
            {
                sprintf(errorMessage, "Out of memory error.");
                return 1;
            }
            
            // find the error of the model when applied to all the points in the
            // consensus set
            int error = getModelError(consensusSet, consensusSetSize, model, distanceThreshold);            

            // if the error of current model is less the the previous best error,
            // set current model plane as the best model plane, set the current
            // error as the best error, and indicate that we have found at least 
            // one model plane to fit the observation data
            if (error < bestError) 
            {
                gsl_vector_memcpy(bestModel, model);
                bestError = error;
                
                foundModel = true;
            }
        }
        
        iterations++;
    }
    
    // if RANSAC did not find a single model plane that fits at least numRequiredInliers
    // number of points in the observation set, then we did not find a good model
    // for the data set (indicating that data set may be invalid), so return error
    if (!foundModel)
    {
        sprintf(errorMessage, "Cannot find fitted model for 3D triangulated points.\n");
        return 1;
    }
    
    // go through the set of valid observation points, and if the point does not
    // fit the best fit model plane determined by RANSAC, we set that point as
    // invalid
    for (int i = 0; i < numPoints; i++)
    {
        if (validFeatureIndicator[i])
        {
            if (!fitsModel(pointCloud[i], bestModel, distanceThreshold)) 
            {
                validFeatureIndicator[i] = '\0';
            }
        }
    }
    
    // cleanup
    gsl_vector_free(bestModel);
    free(maybeInliers);
    free(consensusSet);
    gsl_vector_free(model);    
    
    return 0;
}

