/* File: features.cpp
 *
 * Description: This file contains the program that takes 2 images (from 2 calibrated
 *     cameras) as input, allows the user to select regions of interest in camera 
 *     1 image, find features within them, matches the features from the camera 
 *     2 image, and write the results to a file (to be used to triangulation of
 *     the features).
 *
 * Author: Ming Guo
 * Created: 7/8/11
 */

#include "wing.h"
#include "features.h"

using namespace cv;
using namespace std;

#define STATUS_FAILED(x) (x != 0)

#define MAX_FEATURES 1000000
#define MAX_POLYGONS 50
#define MAX_POLYGON_POINTS 500
#define DEFAULT_SURF_THRESHOLD 400.0
#define MAGNIFIED_ROI_WIDTH 1024

// These are variables that we have to define globally, because they are set in the
// mouse callback function for the user selection of areas of interest, and you 
// cannot pass in more than one variable to the mouse callback function.
IplImage *displayImage;
CvPoint point;
int state;

CvPoint **polygonsOfInterest;
int numPolygons;
int numPointsInPolygons[50];

CvSeq *curPolygonContour;
CvMemStorage *curPolygonContourStorage;

// Cyan lines for selection screens
// OpenCV assumes BGR order for color, instead of RGB
const CvScalar selectionLineColor = cvScalar(UCHAR_MAX, UCHAR_MAX, 0);

const char *boundingBoxSelectionWindowName;
const char *polygonsSelectionWindowName;


/* Function: convertToUint8
 * 
 * Description: creates an 8 bit unsigned int version of input image. If input 
 *     image is already 8 bit unsigned int, set output as copy of input image.
 * 
 * Parameters:
 *     image: Input image to be converted to 8 bit unsigned int.
 *     imageUint8: Output 8-bit unsigned int version of input image.
 * 
 * Returns: 0 on success, error code on error
 */
int convertToUint8(
    _In_ IplImage *image,
    _Out_ IplImage *imageUint8)
{
    if (imageUint8->depth != IPL_DEPTH_8U)
    {
        return IMAGE_DEPTH_ERROR;
    }
    
    // if input image is already 8 bit unsigned int, return copy
    if (image->depth == IPL_DEPTH_8U)
    {
        cvCopy(image, imageUint8);
    }
    
    else
    {
        int height = image->height;
        int width = image->width;
        
        // get maximum pixel value of input image for scaling purposes
        double maxVal = 0;
        cvMinMaxLoc(image, NULL, &maxVal);
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double pixelValue;
                
                // retrieve each pixel value from input image imageData field differently 
                // depending on depth of image
                if (image->depth == IPL_DEPTH_16U)
                {    
                    pixelValue = (double)(((unsigned short *)(image->imageData + image->widthStep*y))[x]);
                }
                else if (image->depth == IPL_DEPTH_32F)
                {
                    pixelValue = (double)(((float *)(image->imageData + image->widthStep*y))[x]);
                }
                else if (image->depth == IPL_DEPTH_64F)
                {
                    pixelValue = (double)(((double *)(image->imageData + image->widthStep*y))[x]);
                }
                else
                {
                    return IMAGE_DEPTH_ERROR;
                }
                
                // set the corresponding pixel in the 8 bit unsigned int image to the 
                // scaled value of the original pixel
                ((unsigned char*)(imageUint8->imageData + imageUint8->widthStep*y))[x] = (unsigned char)(pixelValue / maxVal * UCHAR_MAX);
            }
        }
    }
    
    return 0;
}

/* Function: drawContours
 * 
 * Description: Draws contours on an image. Used for debugging purposes.
 * 
 * Parameters:
 *     contoursImage: Input image to have contours drawn on it
 *     contours: contours to draw on the image
 */
void drawContours(
    _InOut_ IplImage *contoursImage,
    _In_ CvSeq *contours)
{
    // We specify different colors for the contours depending on whether it is an 
    // external or internal contour
    CvScalar externalContourColor = cvScalar(255);
    CvScalar internalContourColor = cvScalar(0);

    // Max level of contours to draw. 1 means only the external contours are drawn.
    int maxLevel = 1;
    int lineThickness = 1;
    int lineType = CV_AA;

    // Draw contours on top of image.
    cvDrawContours(contoursImage, contours, externalContourColor, internalContourColor, maxLevel, lineThickness, lineType);  
}


/* Function: drawCirclesAroundFeatures
 * 
 * Description: Draws circles on the input image around the input features. Used
 *     for debugging purposes.
 * 
 * Parameters:
 *     image: Input image to have features circled.
 *     features: Input features to circle on image.
 *     numFeatures: Number of features in features array.
 */
void drawCirclesAroundFeatures(
    _InOut_ IplImage *image,
    _In_ CvPoint2D32f *features,
    _In_ int numFeatures)
{
    int radius = 1;
    CvScalar color = cvScalar(UCHAR_MAX);
    int thickness = 1;
    int lineType = CV_AA;
    
    for (int i = 0; i < numFeatures; i++)
    {
        cvCircle(image, cvPoint((int) features[i].x, (int) features[i].y), radius, color, thickness, lineType);
    }    
}

/* Function: subtractBackground
 *
 * Description: Subtracts background of input image by testing if a pixel lies 
 *     within input contours using cvPointPolygonTest, and setting the pixel value
 *     to 0 if not
 *
 * Parameters:
 *     backgroundSubtractedImage: Image to subtract the background from.
 *     contours: contours surrounding objects of interest in image
 */
int subtractBackground(_InOut_ IplImage *image, 
                       _In_ CvSeq *contours)
{  
    if (image->depth != IPL_DEPTH_8U)
    {
        return IMAGE_DEPTH_ERROR;
    }
    
    int width = image->width;
    int height = image->height;
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // create CvPoint2D32f for current pixel
            CvPoint2D32f point = cvPoint2D32f((double)x, (double)y);
            
            // Now contour points to the first contour in the list
            CvSeq *contour = contours; 
            bool inContour = false;
        
            // We loop through the list of external contours
            while (contour)
            {
                // Call cvPointPolygonTest function to see if point lies or within 
                // current contour. Setting third parameter to false indicates that 
                // we do not care about distance from point to nearest contour edge.
                if (cvPointPolygonTest(contour, point, false) >= 0)
                {
                    inContour = true;
                    break;
                }
                
                // Once we're done with this contour, we need to move to the 
                // following contour that's not included in the one we just processed.
                // This is pointed to by the h_next (horizontal next) pointer. 
                contour = (CvSeq *)(contour->h_next);
            }

            if (!inContour)
            {
                ((unsigned char*)(image->imageData + image->widthStep*y))[x] = 0;
            }
        }
    }
    
    return 0;
}

/* Function: findMatches
 * 
 * Description: Takes in camera 1 and camera 2 images of the same frame, as well
 *     as the keypoints of previously identified features for both images, found
 *     by either SURF or SIFT feature detector. It matches the features of camera 
 *     2 with the features of camera 1, and outputs those features into a CvPoint2D32f 
 *     array. If the feature detector is not identified as SURF or SIFT, an error
 *     is thrown.
 * 
 * Parameters:
 *     camera1: Camera 1 IplImage
 *     camera2: Camera 2 IplImage
 *     camera1KeyPoints: a vector of KeyPoints containing the features found by
 *         SURF or SIFT in the camera 1 image.
 *     camera2KeyPoints: a vector of KeyPoints containing the features found by
 *         SURF or SIFT in the camera 2 image.
 *     camera2Features: a CvPoint2D32f array containing the camera 2 features
 *         matched to the camera 1 features, with corresponding indices
 *     numFeatures: number of camera 1 features to match with camera 2 features
 *     featureDetector: either SURF or SIFT
 * 
 * Returns: 0 on success, error code on error
 */
int findMatches(
    _In_ IplImage *camera1,
    _In_ IplImage *camera2,
    _In_ vector<KeyPoint> &camera1KeyPoints,
    _In_ vector<KeyPoint> &camera2KeyPoints,
    _Out_ CvPoint2D32f *camera2Features,
    _In_ int numFeatures,
    _In_ char *featureDetector)
{
    // matrices for the feature descriptions for cameras 1 and 2 features
    Mat camera1Descriptors, camera2Descriptors;
    
    DescriptorExtractor *extractor;
    
    // create a SIFT or SURF feature descriptor extractor depending on what the
    // user-specified detector is
    if (!strcasecmp(featureDetector, SIFT_FEATURE_DETECTOR) ||
        !strcasecmp(featureDetector, SPEEDSIFT_FEATURE_DETECTOR))
    {    
        extractor = new SiftDescriptorExtractor(SIFT::DescriptorParams::GET_DEFAULT_MAGNIFICATION(), true, true, 4, 24, -1, 0);
    }
    else if (!strcasecmp(featureDetector, SURF_FEATURE_DETECTOR))
    {
        extractor = new SurfDescriptorExtractor(4, 24, false);
    }
    else
    {
        return INVALID_FEATURE_DETECTOR_ERROR;
    }
    
    // extract the feature descriptors from the features
    extractor->compute(camera1, camera1KeyPoints, camera1Descriptors);
    extractor->compute(camera2, camera2KeyPoints, camera2Descriptors);
    
    BruteForceMatcher<L2<float> > matcher;
    vector<DMatch> matches;
    
    // match camera 2 features with camera 1 features using the brute
    // force matcher
    matcher.match(camera1Descriptors, camera2Descriptors, matches);
    
    int i = 0;
    
    // fill in the CvPoint2D32f array with the matched camera 2 keypoints
    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); it++) 
    {
        KeyPoint c2Match = camera2KeyPoints.at(it->trainIdx);
        
        camera2Features[i].x = c2Match.pt.x;
        camera2Features[i].y = c2Match.pt.y;
        
        i++;
    }
    
    return 0;
}

/* Function: findFeatures
 * 
 * Description: Takes in camera 1 and camera 2 images of the same frame, and finds
 *     features within them using the user-specified feature detector. Stores the
 *     resulting keypoints in output vectors.
 * 
 * Parameters:
 *     camera1: Camera 1 IplImage
 *     camera2: Camera 2 IplImage
 *     camera1KeyPoints: a vector of KeyPoints containing the features found by
 *         the user-specified feature detector in the camera 1 image.
 *     camera2KeyPoints: a vector of KeyPoints containing the features found by
 *         the user-specified feature detector in the camera 2 image.
 *     featureDetector: string that specifies which feature detector to use
 * 
 * Returns: 0 on success, error code on error
 */
int findFeatures(
    _In_ IplImage *camera1,
    _In_ IplImage *camera2,
    _Out_ vector<KeyPoint> &camera1KeyPoints,
    _Out_ vector<KeyPoint> &camera2KeyPoints,
    _In_ char *featureDetector)
{
    FeatureDetector *fd = 0;
    
    // create the feature detector depending on the featureDetector parameter
    if (!strcasecmp(featureDetector, FAST_FEATURE_DETECTOR))
    {
        fd = new FastFeatureDetector(1 /*threshold*/, 
                                     true /*nonmaxSuppression*/);
        
        fd = new PyramidAdaptedFeatureDetector(fd, 5);
    }
    else if (!strcasecmp(featureDetector, GFTT_FEATURE_DETECTOR))
    {                    
        fd = new GoodFeaturesToTrackDetector(MAX_FEATURES /*maxCorners*/, 
                                             0.001 /*qualityLevel*/, 
                                             0.1 /*minDistance*/, 
                                             3 /*blockSize*/, 
                                             false /*useHarrisDetector*/, 
                                             0.04 /*k*/);
        
        fd = new PyramidAdaptedFeatureDetector(fd, 5);
    }
    else if (!strcasecmp(featureDetector, SIFT_FEATURE_DETECTOR) ||
             !strcasecmp(featureDetector, SPEEDSIFT_FEATURE_DETECTOR))
    {
        fd = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD()/4,
                                     SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD(),
                                     4, /*nOctaves (def = 4)*/
                                     24, /*nOctaveLayers (def = 3)*/
                                     -1, /*firstOctave (def = -1)*/
                                     0 /*firstAngle (def = 0)*/);
    }
    else if (!strcasecmp(featureDetector, SURF_FEATURE_DETECTOR))
    {
        fd = new SurfFeatureDetector(DEFAULT_SURF_THRESHOLD/8,
                                     4 /*octaves*/, 
                                     24 /*octave_layers*/);
    }
    else
    {
        return INVALID_FEATURE_DETECTOR_ERROR;
    }
    
    // detect features for the camera 1 and 2 images
    fd->detect(camera1, camera1KeyPoints);
    fd->detect(camera2, camera2KeyPoints);
    
    return 0;
}

/* Function: calculateOpticFlowForFeatureSet
 * 
 * Description: Calculates the optic flow using two frames and a set of features
 *     found in the first frame. Outputs the matching feature points found in the 
 *     second frame into a CvPoint2D32f array, as well as a char array indicating 
 *     which features from the first frame were found in the second frame. Calculated 
 *     using the OpenCV cvCalcOpticalFlowPyrLK function.
 * 
 * Parameters:
 *     frame1: Input image for the first frame.
 *     frame2: Input image for the second frame.
 *     numFeatures: Number of features found in the first frame.
 *     f1Features: CvPoint2D32f array that contains the feature points found in
 *         the first frame.
 *     f2Features: Empty CvPoint2D32f array that function populates with features
 *         points from the second frame that match features from the first frame.
 *     featuresFound: Empty char array where an element is set to 1 by function 
 *         if the flow for the corresponding feature has been found, 0 otherwise.
 * 
 * Returns: 0 on success, 1 on error.
 */
int calculateOpticFlowForFeatureSet(
    _In_ IplImage *frame1, 
    _In_ IplImage *frame2,
    _In_ int numFeatures,
    _In_ CvPoint2D32f *f1Features,
    _Out_ CvPoint2D32f *f2Features,
    _Out_ char *featuresFound)
{    
    // size of the search window of each pyramid level
    CvSize windowSize = cvSize(3,3);
    
    // maximum pyramid level
    int maxPyramidLevel = 5;
    
    // Optional parameter of float array containing the error in optical flow 
    // between corresponding features of the first and second frames.
    float *featuresError = NULL;
    
    // Create termination criteria that tells the algorithm to stop when it has 
    // either done the maximum number of iterations or when its accuracy is better 
    // than episilon.
    int maxIterations = 20;
    double episilon = 0.3;

    CvTermCriteria terminationCriteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIterations, episilon);
    
    // miscellaneous flags we can set for cvCalcOpticalFlowPyrLK function
    int flags = 0;

    // allocate images used as workspace for cvCalcOpticalFlowPyrLK calculations
    IplImage *f1Pyramid = cvCreateImage(cvSize(frame1->width, frame1->height), IPL_DEPTH_8U, 1);
    
    if (f1Pyramid == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    IplImage *f2Pyramid = cvCreateImage(cvSize(frame1->width, frame1->height), IPL_DEPTH_8U, 1);
    
    if (f2Pyramid == NULL)
    {
        cvReleaseImage(&f1Pyramid);
        return OUT_OF_MEMORY_ERROR;
    }

    // Run cvCalcOpticalFlowPyrLK function using frame 1, frame 2, and feature 
    // set found in frame 1, outputting matching frame 2 features to 'f2Features'. 
    // Also outputs to 'featuresFound' whether corresponding frame 1 feature was 
    // found in frame 2.
    cvCalcOpticalFlowPyrLK(frame1, frame2, f1Pyramid, f2Pyramid, f1Features, f2Features, numFeatures, windowSize, maxPyramidLevel, featuresFound, featuresError, terminationCriteria, flags);
    
    // cleanup
    cvReleaseImage(&f1Pyramid);
    cvReleaseImage(&f2Pyramid);
    
    return 0;
}

/* Function: drawFlowArrows
 * 
 * Description: Draw a flow line with arrows between two given points. Used for
 *     debugging purposes.
 * 
 * Parameters:
 *     opticFlowImage: Input image to overlay the optic flow field over.
 *     p: Point where the flow line starts.
 *     q: Point where the flow line ends.
 *     minFlowLength: minimum length of flow lines to draw
 *     lengtheningFactor: factor by which to lengthen the flow lines for visualization
 *         purposes
 *     arrorScale: factor by which to scale the length of the flow line arrows
 */
void drawFlowArrows(
    _InOut_ IplImage *opticFlowImage,
    _In_ CvPoint p,
    _In_ CvPoint q,
    _In_ int minFlowLength,
    _In_ int lengtheningFactor,
    _In_ int arrowScale)
{
    // get the length of the flow
    double flowLength = sqrt(pow(p.y - q.y, 2) + pow(p.x - q.x, 2));    
    
    // only draw the flow lines if they are greater than a minimum length
    if (flowLength > minFlowLength)
    {
        double angle = atan2((double)(p.y - q.y), (double)(p.x - q.x));
        
        q.x = (int) (p.x - lengtheningFactor * flowLength * cos(angle));
        q.y = (int) (p.y - lengtheningFactor * flowLength * sin(angle));
        
        // set parameters for drawing the flow lines
        int lineThickness = 1;
        CvScalar lineColor = cvScalar(UCHAR_MAX);
        int lineType = CV_AA;
        
        // draw the flow line over the input image
        cvLine(opticFlowImage, p, q, lineColor, lineThickness, lineType);
        
        // Draw arrow tips at the end of the flow line for effect.        
        p.x = (int) (q.x + arrowScale * cos(angle + M_PI / 4));
        p.y = (int) (q.y + arrowScale * sin(angle + M_PI / 4));
        cvLine(opticFlowImage, p, q, lineColor, lineThickness, lineType);
        
        p.x = (int) (q.x + arrowScale * cos(angle - M_PI / 4));
        p.y = (int) (q.y + arrowScale * sin(angle - M_PI / 4));
        cvLine(opticFlowImage, p, q, lineColor, lineThickness, lineType);
    }
}

/* Function: drawFlowFieldFeatures
 * 
 * Description: Draw the optic flow field of detected features between two frames 
 *     over the input image. Used for debugging purposes.
 * 
 * Parameters:
 *     opticFlowImage: Input image to overlay the optic flow field over.
 *     numFeatures: Number of features found in the first frame.
 *     f1Features: CvPoint2D32f array that contains the feature points found in
 *         the first frame.
 *     f2Features: CvPoint2D32f array that contains the feature points found in
 *         the second frame.
 *     featuresFound: Char array where an element has been set to 1 if the flow 
 *         for the corresponding feature has been found, 0 otherwise.
 */
void drawFlowFieldFeatures(
    _InOut_ IplImage *opticFlowImage, 
    _In_ int numFeatures,
    _In_ CvPoint2D32f *f1Features,
    _In_ CvPoint2D32f *f2Features,
    _In_ char *featuresFound)
{
    // loop through each feature found in frame 1
    for(int i = 0; i < numFeatures; i++)
    {
        // only draw flow line if corresponding feature was found in frame 2
        if (featuresFound[i])
        {
            // 'p' is the point where the flow line begins (location of the feature
            // in frame 1). 'q' is the point where the flow line stops (location
            // of the feature in frame 2).
            CvPoint p,q;
            p.x = (int) f1Features[i].x;
            p.y = (int) f1Features[i].y;
            q.x = (int) f2Features[i].x;
            q.y = (int) f2Features[i].y;

            // only draw flow lines that are larger than the specified minimum 
            // flow length
            int minFlowLength = 2;
            
            // the flow lines may be too short for a good visualization due to 
            // high framerate between the two frames, so we can lengthen them by 
            // a lengthening factor
            int lengtheningFactor = 1;
             
            // scaling for the arrown tips so that they proportional to the flow 
            // lines
            int arrowScale = 5;
            
            // draw the flow line with arrows
            drawFlowArrows(opticFlowImage, p, q, minFlowLength, lengtheningFactor, arrowScale);
        }
    }
}

/* Function: areFeaturesInContour
 * 
 * Description: Check if features lie within specified single contour. If not, 
 *     set the corresponding value for the feature in the 'validFeatureIndicator' 
 *     char array to 0.
 * 
 * Parameters:
 *     features: array of features to check if in contours
 *     numFeatures: number of features in array
 *     validFeatureIndicator: char array indicating whether a feature is valid
 *         or not. Value for corresponding feature is set to 0 if feature is not
 *         within contours.
 *     contour: single contour to check if features are inside
 */
int areFeaturesInContour(
    _In_ CvPoint2D32f *features2D,
    _In_ int numFeatures,
    _Out_ char *validFeatureIndicator,
    _In_ CvSeq *contour)
{
    int numValidFeatures = 0;
    
    // loop through each feature
    for(int i = 0; i < numFeatures; i++)
    {
        // Call cvPointPolygonTest function to see if point lies on or within 
        // current contour. Setting third parameter to false indicates that 
        // we do not care about distance from point to nearest contour edge.
        if (cvPointPolygonTest(contour, features2D[i], false) >= 0)
        {
            validFeatureIndicator[i] = '1';
            numValidFeatures++;
        }
        else
        {
            validFeatureIndicator[i] = '\0';
        }
    }
    
    return numValidFeatures;
}

/* Function: removeInvalidFeatures
 * 
 * Description: Create a CvPoint2D32f arrays of features with the invalid features
 *     removed.
 * 
 * Parameters:
 *     featuresForImageSet: Array where each element is a feature point array for
 *         a different image.
 *     numImages: Number of sets of image features.
 *     validFeatureIndicator: char array indicating whether a feature is valid
 *         or not. Value for corresponding feature is set to 0 if the optic flow 
 *         for a feature is smaller than the minimum length.
 *     numFeatures: number of features in each feature array.
 * 
 * Returns: 0 on success, 1 on error.
 */
int removeInvalidFeatures(
    _InOut_ CvPoint2D32f **featuresForImageSet,
    _In_ int numImages,
    _In_ char *validFeatureIndicator, 
    _InOut_ int *numFeatures)
{
    // create a new array for image feature sets for valid features
    CvPoint2D32f **validFeaturesForImageSet = (CvPoint2D32f **)malloc(numImages * sizeof(CvPoint2D32f *));
    
    if (validFeaturesForImageSet == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    for (int i = 0; i < numImages; i++)
    {
        // allocate memory for feature set for each image
        validFeaturesForImageSet[i] = (CvPoint2D32f *)malloc(*numFeatures * sizeof(CvPoint2D32f));
    
        if (validFeaturesForImageSet[i] == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
    }
    
    int numValidFeatures = 0;
    
    // loop through each feature in set of all features
    for (int i = 0; i < *numFeatures; i++)
    {
        // if the feature is valid, add the feature to the valid feature array
        // for each image feature set
        if (validFeatureIndicator[i])
        {
            for (int j = 0; j < numImages; j++)
            {
                validFeaturesForImageSet[j][numValidFeatures] = featuresForImageSet[j][i];
            }
                
            numValidFeatures++;
        }
    }
    
    // copy the sets of valid features into the input feature sets array
    for (int i = 0; i < numImages; i++)
    {
        memcpy(featuresForImageSet[i], validFeaturesForImageSet[i], numValidFeatures*sizeof(CvPoint2D32f));
    }
    
    // set the number of features to the number of valid features
    *numFeatures = numValidFeatures;
    
    // cleanup
    for (int i = 0; i < numImages; i++)
    {
        free(validFeaturesForImageSet[i]);
    }
    
    free(validFeaturesForImageSet);
    
    return 0;
}

/* Function: getValidFeatures
 * 
 * Description: Given vectors of feature keypoints for camera 1 and 2 images,
 *     first find the features from camera 1 that lie within the specified
 *     contour, then find the matching features in camera 2 for those in-contour
 *     features.
 * 
 * Parameters:
 *     camera1: Camera 1 IplImage
 *     camera2: Camera 2 IplImage
 *     camera1KeyPoints: a vector of KeyPoints containing the features found by
 *         the user-specified feature detector in the camera 1 image.
 *     camera2KeyPoints: a vector of KeyPoints containing the features found by
 *         the user-specified feature detector in the camera 2 image.
 *     c1Features: CvPoint2D32f array containing camera 1 features
 *     c2Features: CvPoint2D32f array containing camera 2 features
 *     numFeatures: number of features
 *     polygonContour: contour contaning points of interest in camera 1
 *     featureDetector: string that specifies which feature detector to use
 * 
 * Returns: 0 on success, 1 on error.
 */
int getValidFeatures(
    _In_ IplImage *camera1, 
    _In_ IplImage *camera2,
    _In_ vector<KeyPoint> &camera1KeyPoints,
    _In_ vector<KeyPoint> &camera2KeyPoints,
    _InOut_ CvPoint2D32f *c1Features, 
    _Out_ CvPoint2D32f *c2Features,
    _InOut_ int *numFeatures, 
    _In_ CvSeq *polygonContour,
    _In_ char *featureDetector)
{            
    // char array indicating whether a feature is valid or not
    char validFeatureIndicator[MAX_FEATURES]; 
    
    // filter for features that lie within contour of objects of interest
    areFeaturesInContour(c1Features, *numFeatures, validFeatureIndicator, polygonContour);
    
    // create and fill in a vector of keypoints containing in-contour features
    vector<KeyPoint> camera1KeyPointsInContours;
    
    for (int i = 0; i < *numFeatures; i++) 
    {
        if (validFeatureIndicator[i])
        {
            camera1KeyPointsInContours.push_back(camera1KeyPoints.at(i));   
        }
    }
    
    CvPoint2D32f *featuresForImageSet[1] = {c1Features};

    // remove invalid features from feature set of camera 1
    if (STATUS_FAILED(removeInvalidFeatures(featuresForImageSet, 1, validFeatureIndicator, *(&numFeatures))))
    {
        return OUT_OF_MEMORY_ERROR;
    }    
    
    // Check to see if the frames from the first and second camera are of the same
    // dimensions (height and width). If not, pad the first frames of the first 
    // and second cameras so that they are, and use the padded images to find the 
    // shared features between the two images.
    if ((camera1->width != camera2->width) || (camera1->height != camera2-> height))
    {
        // get the larger width and height of the two images
        int resizedWidth = camera1->width;
        
        if (camera1->width < camera2->width)
        {
            resizedWidth = camera2->width;
        }
        
        int resizedHeight = camera1->height;
        
        if (camera1->height < camera2->height)
        {
            resizedHeight = camera2->height;
        }

        // create padded copies of the two images using 'resizedWidth' and 
        // 'resizedHeight'
        IplImage *camera1Resized = cvCreateImage(cvSize(resizedWidth, resizedHeight), camera1->depth, camera1->nChannels);
        cvCopyMakeBorder(camera1, camera1Resized, cvPoint(0,0), IPL_BORDER_CONSTANT, cvScalar(0));

        IplImage *camera2Resized = cvCreateImage(cvSize(resizedWidth, resizedHeight), camera2->depth, camera2->nChannels);
        cvCopyMakeBorder(camera2, camera2Resized, cvPoint(0,0), IPL_BORDER_CONSTANT, cvScalar(0));

        // if feature detector is FAST or GFTT, use optic flow to get the matching
        // camera 2 features
        if (!strcasecmp(featureDetector, FAST_FEATURE_DETECTOR) || 
            !strcasecmp(featureDetector, GFTT_FEATURE_DETECTOR))
        {
            if (STATUS_FAILED(calculateOpticFlowForFeatureSet(camera1Resized, camera2Resized, *numFeatures, c1Features, c2Features, validFeatureIndicator)))
            {
                return OUT_OF_MEMORY_ERROR;
            }
        }
        // if feature detector is SIFT or SURF, use feature matching to get the
        // matching camera 2 features
        else
        {
            memset(validFeatureIndicator, 1, *numFeatures * sizeof(char));
            
            if (STATUS_FAILED(findMatches(camera1Resized, camera2Resized, camera1KeyPointsInContours, camera2KeyPoints, c2Features, *numFeatures, featureDetector)))
            {
                return INVALID_FEATURE_DETECTOR_ERROR;
            }
        }
        
        cvReleaseImage(&camera1Resized);
        cvReleaseImage(&camera2Resized);
    }
    else 
    {
        if (!strcasecmp(featureDetector, FAST_FEATURE_DETECTOR) || 
            !strcasecmp(featureDetector, GFTT_FEATURE_DETECTOR))
        {
            if (STATUS_FAILED(calculateOpticFlowForFeatureSet(camera1, camera2, *numFeatures, c1Features, c2Features, validFeatureIndicator)))
            {
                return OUT_OF_MEMORY_ERROR;
            }
        }
        else
        {
            memset(validFeatureIndicator, 1, *numFeatures * sizeof(char));
            
            if (STATUS_FAILED(findMatches(camera1, camera2, camera1KeyPointsInContours, camera2KeyPoints, c2Features, *numFeatures, featureDetector)))
            {
                return INVALID_FEATURE_DETECTOR_ERROR;
            }
        }
    }
    
    CvPoint2D32f *featuresForImageSet2[2] = {c1Features, c2Features};

    // remove invalid features from feature sets of both camera images
    if (STATUS_FAILED(removeInvalidFeatures(featuresForImageSet2, 2, validFeatureIndicator, *(&numFeatures))))
    {
        return OUT_OF_MEMORY_ERROR;
    }    
    
    return 0;
}


/* Function: writeFeaturesToFile
 * 
 * Description: writes the cameras 1 and 2 feature points for frames 1 and 2 images
 *     to two different files, in the format:
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
 *     numPolygonContours: number of contours
 *     contourNames: names of the contours
 *     c1Features: valid features found for camera 1
 *     c2Features: valid features found for camera 2
 *     numValidFeatures: number of valid features found
 *     outputFilename: name of the output file to write 2D features to
 * 
 * Returns: 0 on success, error code on error.
 */
int writeFeaturesToFile(
    _In_ int numPolygonContours,
    _In_ char **contourNames,
    _In_ CvPoint2D32f **c1Features,
    _In_ CvPoint2D32f **c2Features,
    _In_ int *numFeaturesInContours,
    _In_ char *outputFilename)
{
    // open the output files for writing
    FILE *outputFile = fopen(outputFilename, "w");
    
    if (outputFile == NULL)
    {
        return OUTPUT_FILE_OPEN_ERROR;
    }
    
    fprintf(outputFile, "%d\n", numPolygonContours);
    fprintf(outputFile, "camera_1\n");
    
    for (int i = 0; i < numPolygonContours; i++)
    {  
        fprintf(outputFile, "%s %d\n", contourNames[i], numFeaturesInContours[i]);
    
        for (int j = 0; j < numFeaturesInContours[i]; j++)
        {
            fprintf(outputFile, "%f %f\n", c1Features[i][j].x, c1Features[i][j].y);
        }
    }
    
    fprintf(outputFile, "camera_2\n");
    
    for (int i = 0; i < numPolygonContours; i++)
    {  
        fprintf(outputFile, "%s %d\n", contourNames[i], numFeaturesInContours[i]);
    
        for (int j = 0; j < numFeaturesInContours[i]; j++)
        {
            fprintf(outputFile, "%f %f\n", c2Features[i][j].x, c2Features[i][j].y);
        }
    }
    
    // close output files
    fclose(outputFile);
    
    return 0;
}

/* Function: writeContourVerticesToFile
 * 
 * Description: writes the vertices of the contour (going clockwise) to an output 
 *     file, in the format:
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
 *     polygonContours: CvSeq specifying the vertices of the contours
 *     numPolygonContours: number of ROIs
 *     contourNames: user-specified names for the contours
 *     outputFilename: name of the output file to write contour vertices to
 * 
 * Returns: 0 on success, 1 on error.
 */
int writeContourVerticesToFile(
    _In_ CvSeq *polygonContours,
    _In_ int numPolygonContours,
    _In_ char **contourNames,
    _In_ char *outputFilename)
{
    // open the output file for writing
    FILE *outputFile = fopen(outputFilename, "w");
    
    if (outputFile == NULL)
    {
        return OUTPUT_FILE_OPEN_ERROR;
    }
    
    // write the number of ROIs to file
    fprintf(outputFile, "%d\n", numPolygonContours);
    
    CvSeq *polygonContour = polygonContours;
    int polygonCount = 0;
    
    while (polygonContour)
    {
        // print the contour name and the number of vertices in contour
        fprintf(outputFile, "%s %d\n", contourNames[polygonCount], polygonContour->total);
        
        for (int i = 0; i < polygonContour->total; i++)
        {
            // print each vertex (in a clockwise order)
            CvPoint* contourVertex = (CvPoint *)cvGetSeqElem(polygonContour, i);
            fprintf(outputFile, "%d %d\n", contourVertex->x, contourVertex->y);
        }
        
        polygonContour = (CvSeq *)(polygonContour->h_next);
        polygonCount++;
    }
    
    fclose(outputFile);
    
    return 0;
}
 
/* Function: dragBoundingBoxMouseHandler
 * 
 * Description: Callback function to allow user to select a region of interest 
 *     in the input image by dragging a bounding box.
 * 
 * Parameters:
 *     event: code representing what mouse action occured
 *     x: x coordinate of mouse position
 *     y: y coordinate of mouse position
 *     flags: optional flags for callback function
 *     param: optional parameter for callback function, here used to store the
 *         image where region of interest will be set
 */
void dragBoundingBoxMouseHandler(
    _In_ int event, 
    _In_ int x, 
    _In_ int y, 
    _In_ int flags, 
    _InOut_ void *param)
{    
    IplImage *image = (IplImage *) param;
    
    int lineThickness = 1;
    int lineType = CV_AA;
    
    // user press down left button, start dragging from that mouse position
    if ((event == CV_EVENT_LBUTTONDOWN) && !state)
    {
        point = cvPoint(x, y);
        state = 1;
    }
 
    // if drage state is set (left mouse button is held down), then let user drag 
    // the bounding box to desired dimensions
    if ((event == CV_EVENT_MOUSEMOVE) && (state == 1))
    {
        IplImage *tempDisplayImage = cvCloneImage(displayImage);
               
        cvRectangle(tempDisplayImage, point, cvPoint(x, y), selectionLineColor, lineThickness, lineType);
        cvShowImage(boundingBoxSelectionWindowName, tempDisplayImage);
        
        cvReleaseImage(&tempDisplayImage);
    }
 
    // When user releases left button, set the region of interest in the input
    // image. Set the drag to 2 so that user can't re-drag until they manually
    // reset the ROI.
    if ((event == CV_EVENT_LBUTTONUP) && (state == 1))
    {
        // make sure that the rectangle has an area greater than 0
        if ((x - point.x) * (y - point.y) > 0)
        {
            cvSetImageROI(image, cvRect(point.x, point.y, x - point.x, y - point.y));
        
            cvRectangle(displayImage, point, cvPoint(x, y), selectionLineColor, lineThickness, lineType);
            cvShowImage(boundingBoxSelectionWindowName, displayImage);
        
            state = 2;
        }
        else
        {
            cvShowImage(boundingBoxSelectionWindowName, displayImage);
            state = 0;
        }
    }
     
    // if user click right button, reset the region of interest
    if (event == CV_EVENT_RBUTTONUP)
    {
        cvResetImageROI(image);
        
        displayImage = cvCloneImage(image);
        cvShowImage(boundingBoxSelectionWindowName, displayImage);
        
        state = 0;
    }
}

/* Function: selectRectangleROI
 * 
 * Description: Allows user to select points to create a rectangular regions of 
 *     interest in the input image via dragging a bounding box.
 * 
 * Parameters:
 *     image: input image to select rectangular region of interest in.
 *     windowName: name of the display window
 */
void selectRectangleROI(_In_ IplImage *image, 
                        _In_ const char *windowName)
{      
    boundingBoxSelectionWindowName = windowName;
    
    // create a temporary display image to display for user selecting the ROI,
    // because we want to still display the entire image after the ROI is selected,
    // not just the ROI (which will allow user to re-select ROI if they want to)
    displayImage = cvCloneImage(image);    
    state = 0;    

    cvNamedWindow(boundingBoxSelectionWindowName);
    cvMoveWindow(boundingBoxSelectionWindowName, 0,0);
    
    // set the callback function
    cvSetMouseCallback(boundingBoxSelectionWindowName, dragBoundingBoxMouseHandler, (void *) image);
    
    cvShowImage(boundingBoxSelectionWindowName, displayImage);
    cvWaitKey(0);
    
    cvDestroyWindow(boundingBoxSelectionWindowName);
    cvReleaseImage(&displayImage);
}

/* Function: polygonPointsMouseHandler
 * 
 * Description: Callback function to allow user to select a region of interest 
 *     in the input image by selecting points to form polygon contours around
 *     areas of interest. Can select multiple polygons.
 * 
 * Parameters:
 *     event: code representing what mouse action occured
 *     x: x coordinate of mouse position
 *     y: y coordinate of mouse position
 *     flags: optional flags for callback function
 *     param: optional parameter for callback function, here used to store the
 *         image where region of interest will be set
 */
void polygonPointsMouseHandler(
    _In_ int event, 
    _In_ int x, 
    _In_ int y, 
    _In_ int flags, 
    _In_ void *param)
{   
    IplImage *image = (IplImage *) param;
    
    // parameters for OpenCV drawing functions
    int radius = 1;
    int thickness = 1;
    int lineType = 8;
    bool isClosed = true;
    
    // on upclick of left mouse button, put a polygon vertex at the current mouse
    // position
    if (event == CV_EVENT_LBUTTONUP)
    {
        if (numPolygons < MAX_POLYGONS)
        {
            // here, state indicates whether clicking a point will start a new
            // polygon (state == 0) or add a new vertex in the current polygon
            // (state == 1)
            if (!state)
            {
                // if we are creating a new polygon, allocate memory for the new
                // polygon
                polygonsOfInterest[numPolygons] = (CvPoint *)malloc(MAX_POLYGON_POINTS * sizeof(CvPoint));
                
                if (polygonsOfInterest[numPolygons] == NULL)
                {
                    printf("Features: Out of memory error.\n");
                    exit(1);
                }
                
                // set the number of vertices in the new polygon to 0
                numPointsInPolygons[numPolygons] = 0;
                
                // set state to 1, as we are now adding vertices to an existing
                // polygon
                state = 1;
            }
            
            if (numPointsInPolygons[numPolygons] < MAX_POLYGON_POINTS)
            {
                point = cvPoint(x, y);
                
                // add the point of the mouse click to the current polygon as a
                // new vertex
                polygonsOfInterest[numPolygons][numPointsInPolygons[numPolygons]] = point;        
                numPointsInPolygons[numPolygons]++;
                
                // display the vertex as a small circle on the image
                cvCircle(displayImage, point, radius, selectionLineColor, thickness, lineType);
                
                // draw a line connecting the current vertex to the last vertex,
                // if one exists
                if (numPointsInPolygons[numPolygons] > 1)
                {
                    cvLine(displayImage, point, polygonsOfInterest[numPolygons][numPointsInPolygons[numPolygons]-2], selectionLineColor, thickness, lineType);
                }
            }
        }
        
        cvShowImage(polygonsSelectionWindowName, displayImage);
    }
     
    // on upclick of right mouse button, delete either vertices or entire polygons,
    // depending on the current state
    if (event == CV_EVENT_RBUTTONUP)
    {
        // if still adding vertices to a current polygon (state == 1), delete the
        // last vertex created and set the display image accordingly
        if (state)
        {
            if (numPointsInPolygons[numPolygons] > 0)
            {
                numPointsInPolygons[numPolygons]--;                
                
                displayImage = cvCloneImage(image); 
                cvPolyLine(displayImage, polygonsOfInterest, numPointsInPolygons, numPolygons, isClosed, selectionLineColor, thickness, lineType);
                
                if (numPointsInPolygons[numPolygons] > 0)
                {
                    cvCircle(displayImage, polygonsOfInterest[numPolygons][0], radius, selectionLineColor, thickness, lineType);

                    for (int i = 1; i < numPointsInPolygons[numPolygons]; i++)
                    {
                        cvCircle(displayImage, polygonsOfInterest[numPolygons][i], radius, selectionLineColor, thickness, lineType);
                        cvLine(displayImage, polygonsOfInterest[numPolygons][i], polygonsOfInterest[numPolygons][i-1], selectionLineColor, thickness, lineType);
                    }
                }
            }
            else
            {
                // if we just deleted the last vertex in the current polygon, we
                // remove the polygon from the array of polygons and set the state
                // to 0 to indicate that we are not currently adding vertices to
                // an existing polygon
                free(polygonsOfInterest[numPolygons]);
                state = 0;
            }
        }
        
        // if we are not currently adding vertices to an existing polygon, delete
        // the last polygon created and set the display image and polygon contour
        // linked list accordingly
        if (!state)
        {            
            if (numPolygons > 0)
            {                
                if (curPolygonContour->h_prev)
                {
                    curPolygonContour = curPolygonContour->h_prev;
                    curPolygonContour->h_next = NULL;
                }
                else
                {
                    curPolygonContour = NULL;
                }
                
                numPolygons--;
                
            }
            
            displayImage = cvCloneImage(image);
            cvPolyLine(displayImage, polygonsOfInterest, numPointsInPolygons, numPolygons, isClosed, selectionLineColor, thickness, lineType);
        }
        
        cvShowImage(polygonsSelectionWindowName, displayImage);
    }
}

/* Function: selectPolygonROI
 * 
 * Description: Allows user to select points to create one or more polygon contours
 *     around regions of interest in the input image.
 * 
 * Parameters:
 *     image: input image to select polygon region of interest in.
 *     polygonContours: output parameter to copy the selected polygon contours to
 *     polygonContourStorage: output paramter to store the selected polygon contours
 *     numPolygonContours: number of contours selected
 *     windowName: name of the display window
 * 
 * Returns: 0 on success, 1 on error.
 */
int selectPolygonROI(
    _In_ IplImage *image, 
    _Out_ CvSeq **polygonContours, 
    _Out_ CvMemStorage *polygonContourStorage,
    _Out_ int *numPolygonContours,
    _In_ const char *windowName)
{
    polygonsSelectionWindowName = windowName;
    
    // create a temporary display image to display for user selecting the ROI,
    // so we can draw lines on the image for display purposes.
    displayImage = cvCloneImage(image);
    
    // allocate memory for array of CvPoint arrays to store the vertices of 
    // multiple polygons
    polygonsOfInterest = (CvPoint **)malloc(MAX_POLYGONS * sizeof(CvPoint *));
    
    if (polygonsOfInterest == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    numPolygons = 0;
    
    // create temporary storage and pointer to contours that will be copied to the
    // output contour pointer and storage. This is because we need to declare these
    // variables globally, because they need to be used in the mouse callback
    // function, yet there is no way to pass more than one parameter to the callback 
    // function. We don't want the actual contours to be used in feature detection 
    // to be global, because that is bad coding practice.
    curPolygonContourStorage = cvCreateMemStorage(0);
    
    if (curPolygonContourStorage == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    curPolygonContour = NULL;
    
    // state starts at 0, meaning you are not currently adding vertices to an 
    // existing polygon
    state = 0;
    
    cvNamedWindow(polygonsSelectionWindowName);
    cvMoveWindow(polygonsSelectionWindowName, 0,0);
    
    // set the callback function
    cvSetMouseCallback(polygonsSelectionWindowName, polygonPointsMouseHandler, (void *) image);
    
    cvShowImage(polygonsSelectionWindowName, displayImage);
    
    while (1) 
    {
        // wait for keyboard input
        int key = cvWaitKey(0);
       
        // if "enter" is pressed, quit ROI selection image
        if (key == '\n')
        {
            break;
        }
        
        // if spacebar is pressed, current polygon is completed if currently in
        // polygon-building state. If not in polygon-building state, do nothing.
        else if (key == ' ')
        {
            if (state)
            {   
                // create a temporary CvSeq * to store the polygon contour that
                // was just created
                CvSeq *tempContour = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), curPolygonContourStorage);
                
                
                // retrieve the vertices of the polygon from the polygonsOfInterest
                // array and add the points to the contour
                for (int i = 0; i < numPointsInPolygons[numPolygons]; i++)
                {
                    CvPoint clickedPoint = cvPoint(polygonsOfInterest[numPolygons][i].x, polygonsOfInterest[numPolygons][i].y);
                    cvSeqPush(tempContour, &clickedPoint);
                }
                
                // only add the contour to the linked list of polygon contours
                // if the area inside the contour is greater than 0 (ie. the
                // contour is not a line or point)
                if (fabs(cvContourArea(tempContour)) > 0)
                {
                    if (curPolygonContour)
                    {
                        curPolygonContour->h_next = tempContour;
                        curPolygonContour->h_next->h_prev = curPolygonContour;
                        
                        curPolygonContour = curPolygonContour->h_next;
                    }
                    else
                    {
                        curPolygonContour = tempContour;
                    }
                    
                    numPolygons++;

                    // display the newly created polygon (and all previously created
                    // polygons)
                    displayImage = cvCloneImage(image);
                    
                    int thickness = 1;
                    int lineType = 8;
                    bool isClosed = true;
                    
                    cvPolyLine(displayImage, polygonsOfInterest, numPointsInPolygons, numPolygons, isClosed, selectionLineColor, thickness, lineType);
                    
                    // set the state to 0 to indicate that we are no longer adding
                    // new vertices to an existing polygon
                    state = 0;
                }
            }
        }
        
        // if 'escape' is pressed, clear everything
        else if (key == '\e')
        {
            for (int i = 0; i < numPolygons; i++)
            {
                free(polygonsOfInterest[i]);
            }
            
            numPolygons = 0;            
            curPolygonContour = NULL;
            
            displayImage = cvCloneImage(image);
            
            state = 0;
        }
        
        cvShowImage(polygonsSelectionWindowName, displayImage);
    }
        
    cvDestroyWindow(polygonsSelectionWindowName);    
    cvReleaseImage(&displayImage);
    
    // copy the temporary contours declared in this function to the output contour
    // parameters
    if (curPolygonContour != NULL)
    {
        while (curPolygonContour->h_prev)
        {   
            curPolygonContour = curPolygonContour->h_prev;
        }
    
        *polygonContours = cvCloneSeq(curPolygonContour, polygonContourStorage);
    
        while (curPolygonContour->h_next)
        {
            (*polygonContours)->h_next = cvCloneSeq(curPolygonContour->h_next, polygonContourStorage);
            (*polygonContours)->h_next->h_prev = *polygonContours;
        
            *polygonContours = (*polygonContours)->h_next;
            curPolygonContour = curPolygonContour->h_next;
        }
    
        while ((*polygonContours)->h_prev)
        {
            *polygonContours = (*polygonContours)->h_prev;
        }
    }
    
    *numPolygonContours = numPolygons;
    
    // cleanup
    for (int i = 0; i < numPolygons; i++)
    {
        free(polygonsOfInterest[i]);
    }
            
    free(polygonsOfInterest);    
    cvReleaseMemStorage(&curPolygonContourStorage);
    
    return 0;
}

/* Function: getCorrespondingOriginalImageContoursFromROIContours
 * 
 * Description: Because we select contours from an expanded view of the region
 *     of interest within in an image, we have to convert the coordinates of the
 *     selected vertices to corresponding vertices in the original image. This
 *     function performs those calculations
 * 
 * Parameters:
 *     ROIContours: contour vertices of the enlarged ROI image
 *     imageROI: CvRect representing the size and location within the original
 *         image
 *     resizedROIImageWidth: width of the enlarged ROI image
 *     resizedROIImageHeight: height of the enlarged ROI image
 *     polygonContours: output CvSeq containing the contours scaled to the original 
 *         image
 *     polygonContourStorage: memory to store the polygon contours
 */
void getCorrespondingOriginalImageContoursFromROIContours(
    _In_ CvSeq *ROIContours, 
    _In_ CvRect imageROI, 
    _In_ int resizedROIImageWidth, 
    _In_ int resizedROIImageHeight,
    _Out_ CvSeq **polygonContours,
    _InOut_ CvMemStorage *polygonContourStorage)
{
    // Now contour points to the first contour in the list
    CvSeq *contour = ROIContours;
    *polygonContours = NULL;
    
    // We loop through the list of external contours
    while (contour)
    {
        // create a new contour
        CvSeq *tempContour = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), polygonContourStorage);
        
        // for each vertex in the contour, scale it from the enlarged ROI image to
        // the original image to find the corresponding vertex in the original
        // image
        for (int i = 0; i < contour->total; i++)
        {
            CvPoint *p = CV_GET_SEQ_ELEM(CvPoint, contour, i);
            CvPoint newPoint = cvPoint(imageROI.x + (int)(p->x * ((float)imageROI.width/(float)resizedROIImageWidth)), imageROI.y + (int)(p->y * ((float)imageROI.height/(float)resizedROIImageHeight)));
            
            cvSeqPush(tempContour, &newPoint);
        }
        
        // set tempContour as the next contour, and then set the current contour
        // to the next contour
        if (*polygonContours)
        {
            (*polygonContours)->h_next = tempContour;
            (*polygonContours)->h_next->h_prev = *polygonContours;
            
            *polygonContours = (*polygonContours)->h_next;
        }
        else
        {
            *polygonContours = tempContour;
        }
        
        contour = contour->h_next;
    }
    
    // backtrack so that polygonContours starts at the first contour
    while ((*polygonContours)->h_prev)
    {
        *polygonContours = (*polygonContours)->h_prev;
    }
}

/* Function: flipContoursOverXAxis
 * 
 * Description: Because we select contours from a display image that was flipped 
 *     over the x-axis from the original image, we have to convert the coordinates 
 *     of the selected vertices to corresponding vertices in the original image.
 *     This function performs the flip in place.
 * 
 * Parameters:
 *     polygonContours: contour vertices to flip over x axis
 *     polygonContourStorage: memory to store the polygon contours
 *     imageHeight: height of the original image
 */
void flipContoursOverXAxis(
    _InOut_ CvSeq **polygonContours,
    _In_ CvMemStorage *polygonContourStorage,
    _In_ int imageHeight)
{
    // loop through the contours
    while (1)
    {
        // create a new contour
        CvSeq *tempContour = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), polygonContourStorage);
        
        // for each vertex in the contour, flip it over the x-axis and add it to
        // tempContour
        for (int i = 0; i < (*polygonContours)->total; i++)
        {
            CvPoint *contourVertex = CV_GET_SEQ_ELEM(CvPoint, *polygonContours, i);
            CvPoint xAxisFlippedVertex = cvPoint(contourVertex->x, imageHeight - contourVertex->y);
                        
            cvSeqPush(tempContour, &xAxisFlippedVertex);
        }
        
        // set replace the current contour with tempContour, and break from loop
        // if this is the last contour
        tempContour->h_prev = (*polygonContours)->h_prev;
        tempContour->h_next = (*polygonContours)->h_next;        
        
        if ((*polygonContours)->h_next == NULL)
        {            
            *polygonContours = tempContour;
            break;
        }
                
        tempContour->h_next->h_prev = tempContour;
                
        *polygonContours = tempContour;
        *polygonContours = (CvSeq *)((*polygonContours)->h_next);
    }
    
    // backtrack so that polygonContours starts at the first contour
    while ((*polygonContours)->h_prev)
    {
        *polygonContours = (*polygonContours)->h_prev;
    }
}

/* Function: features
 * 
 * Description: Takes in 2 images from 2 calibrated cameras, allows user to select a 
 *     region to enlarge within the camera 1 image, select the contours of interest 
 *     within that region, find matching features across the 2 cameras that lie 
 *     within those contours, and output the results to a file
 * 
 * Parameters:
 *     camera1: camera 1 image
 *     camera2: camera 2 image
 *     featuresFilename: filename to write the found and matched features to
 *     contoursFilename: filename to write the vertices of the selected contours to
 *     featureDetector: string representing the feature detector to use to find
 *         features
 *     errorMessage: string to output an error message to, on error
 * 
 * Returns: 0 on success, 1 on error.
 */
int features(
    _In_ IplImage *camera1,
    _In_ IplImage *camera2,
    _In_ char *featuresFilename,
    _In_ char *contoursFilename,
    _In_ char *featureDetector,
    _Out_ char *errorMessage)
{   
    // feature detection can only be done on 8-bit unsigned images, so if images
    // are not 8-bit unsigned, convert to 8-bit unsigned
    IplImage *camera1FindAndMatchFeaturesImage = cvCreateImage(cvSize(camera1->width, camera1->height), IPL_DEPTH_8U, 1);
    IplImage *camera2FindAndMatchFeaturesImage = cvCreateImage(cvSize(camera2->width, camera2->height), IPL_DEPTH_8U, 1);
    
    if (camera1FindAndMatchFeaturesImage == NULL || camera2FindAndMatchFeaturesImage == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    if (STATUS_FAILED(convertToUint8(camera1, camera1FindAndMatchFeaturesImage)))
    {
        sprintf(errorMessage, "Invalid image depth.");
        return 1;
    }
    
    if (STATUS_FAILED(convertToUint8(camera2, camera2FindAndMatchFeaturesImage)))
    {
        sprintf(errorMessage, "Invalid image depth.");
        return 1;
    }
    
    IplImage *camera1RGB = cvCreateImage(cvSize(camera1FindAndMatchFeaturesImage->width, camera1FindAndMatchFeaturesImage->height), camera1FindAndMatchFeaturesImage->depth, 3);
    
    if (camera1RGB == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    // convert images from grayscale to BGR, so that we can use colored lines for
    // contour selection
    cvCvtColor(camera1FindAndMatchFeaturesImage, camera1RGB, CV_GRAY2BGR);
    
    // because OpenCV coordinates are different from the coordinates corresponding 
    // to the camera coefficients, the camera images are flipped over the x-axis
    // from the way they are displayed in the video. We flip the display image over 
    // the x-axis so that it is shown the way it looks in the video.
    cvFlip(camera1RGB, NULL, 0);
    
    // select region for magnification
    selectRectangleROI(camera1RGB, "Select camera 1 region to magnify");
    CvRect camera1ROI = cvGetImageROI(camera1RGB);
    
    // resize selected region to be a set width by scaled height
    IplImage *resizedCamera1ROIImage = cvCreateImage(cvSize(MAGNIFIED_ROI_WIDTH, (int)(MAGNIFIED_ROI_WIDTH * ((float)camera1ROI.height/(float)camera1ROI.width))), camera1RGB->depth, camera1RGB->nChannels);
    
    if (resizedCamera1ROIImage == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    cvResize(camera1RGB, resizedCamera1ROIImage, CV_INTER_LINEAR);
        
    cvReleaseImage(&camera1RGB);
       
    int numPolygonContours;
    CvMemStorage *polygonContourStorage = cvCreateMemStorage(0);
    
    if (polygonContourStorage == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    CvSeq *ROIPolygonContours = NULL;
    
    // select the polygon ROIs in which to find features from camera 1 ROI image
    if (STATUS_FAILED(selectPolygonROI(resizedCamera1ROIImage, &ROIPolygonContours, polygonContourStorage, &numPolygonContours, "Select contour(s)")))
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    if (ROIPolygonContours == NULL)
    {
        sprintf(errorMessage, "No contours were selected.");
        return 1;
    }
    
    // get the corresponding vertex points in the original image of the selected
    // contours
    CvSeq *polygonContours;
    getCorrespondingOriginalImageContoursFromROIContours(ROIPolygonContours, 
                                                         camera1ROI, 
                                                         resizedCamera1ROIImage->width, 
                                                         resizedCamera1ROIImage->height, 
                                                         &polygonContours, 
                                                         polygonContourStorage);
    
    cvReleaseImage(&resizedCamera1ROIImage);
    
    // because we selected contours on the x-axis flipped display image, we now
    // need to flip the contour vertices over the x-axis so that they correspond
    // to the original image
    flipContoursOverXAxis(&polygonContours, polygonContourStorage, camera1FindAndMatchFeaturesImage->height);  
    
    // if SPEEDSIFT is selected as the feature detector, black out the background
    // of camera 1 (anything not within the contours). We don't do this for other
    // feature detectors only because by blacking out the background, we will get
    // a set of features during feature finding that is different from the set of
    // features found without the background blacked out
    if (!strcasecmp(featureDetector, SPEEDSIFT_FEATURE_DETECTOR))
    {
        if (STATUS_FAILED(subtractBackground(camera1FindAndMatchFeaturesImage, polygonContours)))
        {
            sprintf(errorMessage, "Invalid image depth.");
            return 1;
        }
    }
    
    // allow user to enter contour names from commandline, in the order that they 
    // selected the contours, and put them in a string array
    char **contourNames = (char **)malloc(numPolygonContours * sizeof(char *));
    
    if (contourNames == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    for (int i = 0; i < numPolygonContours; i++)
    {
        contourNames[i] = (char *)malloc(MAX_CONTOUR_NAME_LENGTH * sizeof(char));
        
        if (contourNames[i] == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        printf("Contour %d name: ", i+1);
            
        if (scanf("%s", contourNames[i]) != 1)
        {
            sprintf(errorMessage, "Could not read from standard input.");
            return 1;
        }
        
        char c;        
        while((c = getchar()) != '\n' && c != EOF);
    }         
    
    // if SPEEDSIFT is selected as the feature detector, allow user to select
    // (using a bounding box) a smaller region within camera 2 image of the
    // region matching the contours selected in the camera 1 image
    if (!strcasecmp(featureDetector, SPEEDSIFT_FEATURE_DETECTOR))
    {        
        IplImage *camera2RGB = cvCreateImage(cvSize(camera2FindAndMatchFeaturesImage->width, camera2FindAndMatchFeaturesImage->height), camera2FindAndMatchFeaturesImage->depth, 3);
        cvCvtColor(camera2FindAndMatchFeaturesImage, camera2RGB, CV_GRAY2BGR);   
        
        cvFlip(camera2RGB, NULL, 0);
        
        selectRectangleROI(camera2RGB, "Select camera 2 region containing contours");
        CvRect camera2ROI = cvGetImageROI(camera2RGB);
                        
        cvReleaseImage(&camera2RGB);
        
        CvSeq *contourBox = cvCreateSeq(CV_SEQ_POLYGON, sizeof(CvSeq), sizeof(CvPoint), polygonContourStorage);
        
        CvPoint upperLeftPoint = cvPoint(camera2ROI.x, camera2ROI.y);
        CvPoint upperRightPoint = cvPoint(camera2ROI.x + camera2ROI.width, camera2ROI.y);
        CvPoint lowerRightPoint = cvPoint(camera2ROI.x + camera2ROI.width, camera2ROI.y + camera2ROI.height);
        CvPoint lowerLeftPoint = cvPoint(camera2ROI.x, camera2ROI.y + camera2ROI.height);
        
        cvSeqPush(contourBox, &upperLeftPoint);
        cvSeqPush(contourBox, &upperRightPoint);
        cvSeqPush(contourBox, &lowerRightPoint);
        cvSeqPush(contourBox, &lowerLeftPoint);        
        
        flipContoursOverXAxis(&contourBox, polygonContourStorage, camera2FindAndMatchFeaturesImage->height);

        // black out the background outside of the rectangle of interest
        if (STATUS_FAILED(subtractBackground(camera2FindAndMatchFeaturesImage, contourBox)))
        {
            sprintf(errorMessage, "Invalid image depth.");
            return 1;
        }
    }
            
    vector<KeyPoint> camera1KeyPoints;
    vector<KeyPoint> camera2KeyPoints;    
    
    // find features in the camera 1 and camera 2 images
    if (STATUS_FAILED(findFeatures(camera1FindAndMatchFeaturesImage, camera2FindAndMatchFeaturesImage, camera1KeyPoints, camera2KeyPoints, featureDetector)))
    {
        sprintf(errorMessage, "Invalid feature detector.");
        return 1;
    }
    
    int numFeaturesFound = camera1KeyPoints.size();    
    
    // allocate memory to store valid features found
    CvPoint2D32f **c1Features = (CvPoint2D32f **)malloc(numPolygonContours * sizeof(CvPoint *));
    CvPoint2D32f **c2Features = (CvPoint2D32f **)malloc(numPolygonContours * sizeof(CvPoint *));
    
    if (c1Features == NULL || c2Features == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    }
    
    for (int i = 0; i < numPolygonContours; i++)
    {
        c1Features[i] = (CvPoint2D32f *)malloc(numFeaturesFound * sizeof(CvPoint2D32f));
        c2Features[i] = (CvPoint2D32f *)malloc(numFeaturesFound * sizeof(CvPoint2D32f));
        
        if (c1Features[i] == NULL || c2Features[i] == NULL)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
    }
    
    int *numFeaturesInContours = (int *)malloc(numPolygonContours * sizeof(int));
    
    if (numFeaturesInContours == NULL)
    {
        sprintf(errorMessage, "Out of memory error.");
        return 1;
    } 
    
    // create CvPoint2D32f for the keypoint features
    for (int i = 0; i < numPolygonContours; i++) 
    {
        for (int j = 0; j < numFeaturesFound; j++)
        {
            c1Features[i][j].x = camera1KeyPoints.at(j).pt.x;
            c1Features[i][j].y = camera1KeyPoints.at(j).pt.y;
        }
        
        numFeaturesInContours[i] = numFeaturesFound;
    }
    
    CvSeq *polygonContour = polygonContours;
    int polygonCount = 0;
    
    // loop through the sequence of contours
    while (polygonContour)
    {           
        // get valid matching in-contour features
        int status = getValidFeatures(camera1FindAndMatchFeaturesImage, 
                                      camera2FindAndMatchFeaturesImage, 
                                      camera1KeyPoints, 
                                      camera2KeyPoints, 
                                      c1Features[polygonCount], 
                                      c2Features[polygonCount], 
                                      &(numFeaturesInContours[polygonCount]), 
                                      polygonContour, 
                                      featureDetector);
        
        if (status == OUT_OF_MEMORY_ERROR)
        {
            sprintf(errorMessage, "Out of memory error.");
            return 1;
        }
        
        if (status == INVALID_FEATURE_DETECTOR_ERROR)
        {
            sprintf(errorMessage, "Invalid feature detector.");
            return 1;
        }
            
        printf("Matching features found in contour %s: %d\n", contourNames[polygonCount], numFeaturesInContours[polygonCount]);
        
        polygonContour = (CvSeq *)(polygonContour->h_next);
        polygonCount++;
    }
    
    // write the valid features found across all cameras to an output file
    if (STATUS_FAILED(writeFeaturesToFile(numPolygonContours, contourNames, c1Features, c2Features, numFeaturesInContours, featuresFilename)))
    {
        sprintf(errorMessage, "Could not open output file.");
        return 1;
    }
    
    // write the vertices of the ROI to a file
    if (STATUS_FAILED(writeContourVerticesToFile(polygonContours, numPolygonContours, contourNames, contoursFilename)))
    {
        sprintf(errorMessage, "Could not open output file.");
        return 1;
    }
    
    // cleanup    
    for (int i = 0; i < numPolygonContours; i++)
    {
        free(c1Features[i]);
        free(c2Features[i]);
        free(contourNames[i]);
    }
    
    free(c1Features);
    free(c2Features);
    free(contourNames);    
    free(numFeaturesInContours);
    
    cvReleaseMemStorage(&polygonContourStorage); 
    
    cvReleaseImage(&camera1FindAndMatchFeaturesImage);
    cvReleaseImage(&camera2FindAndMatchFeaturesImage);
    
    return 0;
}

