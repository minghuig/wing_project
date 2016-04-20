/* File: videos.cpp
 *
 * Description: This file contains the program that can be run on the commandline
 *     and will run features, triangulation, mesh, and flow for 2 videos from
 *     calibrated cameras, given user-inputted parameters. The program will output
 *     all intermediate and final results to file.
 *
 * Author: Ming Guo
 * Created: 1/11/12
 */

#include "engine.h"

#include "videos.h"
#include "features.h"
#include "triangulation.h"
#include "mesh.h"
#include "flow.h"
#include "ransac.h"

#define MAX_FILENAME_LENGTH 500
#define MAX_FILES_IN_DIRECTORY 1000

#define MAX_CONTOURS 100
#define MAX_FEATURES_IN_CONTOUR 1000000

// enum for the different types of image depths to be handled
enum ImageDepth
{
    UINT8,
    UINT16,
    UNSUPPORTED_DEPTH
};

// struct to store input image information
typedef struct
{
    char *imageData;
    int imageWidth;
    int imageHeight;
    int imageWidthStep;
    enum ImageDepth imageDepth;
} ImageInfo;

/* Function: getInputImageDepth
 *
 * Description: Get the depth of the input image.
 *
 * Parameters:
 *     inputImage: The input image, stored in an mxArray.
 * 
 * Returns: The enum of the depth of the input image (UINT8, UINT16, or 
 *     UNSUPPORTED_DEPTH).
 */
enum ImageDepth getInputImageDepth(_In_ const mxArray *inputImageData)
{
     // check what type of data is contained in mxArray of image data
    if (mxIsUint8(inputImageData))
    {
        return UINT8;
    }
    else if (mxIsUint16(inputImageData))
    {
        return UINT16;
    }
    else
    {
        return UNSUPPORTED_DEPTH;
    }
}

/* Function: getInputImageInfo
 *
 * Description: Populates an ImageInfo struct array with image information and data 
 *     passed in from MATLAB.
 *
 * Parameters:
 *     imageInfo: output ImageInfo struct to population with information exacted
 *         by function, using input passed in from MATLAB
 *     imageData: image data passed in from MATLAB
 *     imageDimensions: the dimensions of the image passed in from MATLAB
 *     imagePaddedWidth: the padded width of the image passed in from MATLAB
 * 
 * Returns: 0 on success, error code on error
 */
int getInputImageInfo(
    _InOut_ ImageInfo *imageInfo,
    _In_ mxArray *imageData,
    _In_ mxArray *imageDimensions,
    _In_ mxArray *imagePaddedWidth)
{
    // check for valid values in properties
    if (mxGetNumberOfDimensions(imageData) != 2)
    {
        return IMAGE_INFO_DATA_ERROR;
    }
    
    if (mxGetNumberOfElements(imageDimensions) != 2)
    {
        return IMAGE_INFO_DIMENSIONS_ERROR;
    }
    
    if (mxGetNumberOfElements(imagePaddedWidth) != 1)
    {
        return IMAGE_INFO_PADDED_WIDTH_ERROR;
    }
    
    // set ImageInfo image data
    imageInfo->imageData = (char *)mxGetData(imageData);
    
    // set ImageInfo image depth
    imageInfo->imageDepth = getInputImageDepth(imageData);

    // only supports input depth of 8 or 16 bit unsigned integers
    if (imageInfo->imageDepth == UNSUPPORTED_DEPTH)
    {
        IMAGE_DEPTH_ERROR;
    }
    
    // set ImageInfo image dimensions
    imageInfo->imageWidth = (int) mxGetPr(imageDimensions)[0];
    imageInfo->imageHeight = (int) mxGetPr(imageDimensions)[1];
    
    // set ImageInfo width step
    imageInfo->imageWidthStep = (int) (mxGetScalar(imagePaddedWidth) * ((imageInfo->imageDepth == UINT16) ? sizeof(unsigned short) : sizeof(unsigned char)));
}

/* Function: createIplImageFromImageInfo
 *
 * Description: Populates an IplImage with image information and data stored in 
 *     an ImageInfo struct.
 *
 * Parameters:
 *     image: empty IplImage to populate using values from ImageInfo
 *     imageInfo: ImageInfo struct populated with image info from MATLAB input
 * 
 * Returns: 0 on success, error code on error
 */
int createIplImageFromImageInfo(
    _InOut_ IplImage **image,
    _In_ ImageInfo imageInfo)
{ 
    // create single-channel IplImage with same height, width, and depth as
    // field values in ImageInfo struct
    *image = cvCreateImageHeader(cvSize(imageInfo.imageWidth, imageInfo.imageHeight), (imageInfo.imageDepth == UINT16) ? IPL_DEPTH_16U : IPL_DEPTH_8U, 1);
    
    // allocate memory for IplImage image data
    (*image)->imageData = (char *)malloc(imageInfo.imageWidthStep * imageInfo.imageHeight * sizeof(char));
    
    if ((*image)->imageData == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // copy image data into IplImage
    memcpy((*image)->imageData, imageInfo.imageData, imageInfo.imageWidthStep * imageInfo.imageHeight);        
        
    // set additioinal IplImage info
    (*image)->widthStep = imageInfo.imageWidthStep;
    (*image)->imageDataOrigin = (*image)->imageData;
    
    return 0;
}

/* Function: getVideoFileExtension
 *
 * Description: extracts the file extention from a filename
 *
 * Parameters:
 *     videoFilename: filename of the file
 *     videoExtension: output string for the file extension
 * 
 * Returns: 0 on success, error code on error
 */
int getVideoFileExtension(
    _In_ char *videoFilename,
    _Out_ char *videoExtension)
{
    int videoFilenameLength = strlen(videoFilename);    
    int videoExtensionStartIndex = -1;
    
    // find a period in the filename and store its index
    for (int i = videoFilenameLength - 1; i >= 0 ; i--)
    {
        if (videoFilename[i] == '.')
        {
            videoExtensionStartIndex = i;
        }
    }
        
    // if filename doesn't have a period, then invalid file extension
    if (videoExtensionStartIndex == -1)
    {
        return INVALID_VIDEO_FILE_EXTENSION_ERROR;
    }
    
    int videoExtensionLength = videoFilenameLength - videoExtensionStartIndex;
    
    // copy the remainder of the filename after the period into videoExtension
    for (int i = 0; i < videoExtensionLength; i++)
    {
        videoExtension[i] = videoFilename[videoExtensionStartIndex + i];
    }
    
    // gotta null terminate a C string
    videoExtension[videoExtensionLength] = '\0';
    
    return 0;
}

/* Function: fileExists
 *
 * Description: check if a given filename exists
 *
 * Parameters:
 *     filename: the filename to check the existence of
 * 
 * Returns: true if file exists, false otherwise
 */
bool fileExists(char *filename)
{
    if (FILE *file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    
    return false;
}

/* Function: readContourNamesFromInputFile
 * 
 * Description: reads contour names for contours selected in features.cpp and puts
 *     them into a string array. File must be in the format:
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
 *     numContours: number of contours
 *     contourNames: output string array of contour names
 *     filename: name of the input file
 * 
 * Returns: 0 on success, error code on error.
 */
int readContourNamesFromInputFile(
    _Out_ int *numContours,
    _Out_ char ***contourNames,
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
    
    *contourNames = (char **)malloc(*numContours * sizeof(char *));
    
    if (*contourNames == NULL)
    {
        return OUT_OF_MEMORY_ERROR;
    }
    
    // for each contour, get the name of the contour while checking for the
    // correct file format
    for (int i = 0; i < *numContours; i++)
    {
        int numVertices;
        (*contourNames)[i] = (char *)malloc(MAX_CONTOUR_NAME_LENGTH * sizeof(char));
        
        if ((*contourNames)[i] == NULL)
        {
            return OUT_OF_MEMORY_ERROR;
        }
        
        if (fscanf(contourVerticesFile, "%s %d", (*contourNames)[i], &numVertices) != 2)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }        
        
        for (int j = 0; j < numVertices; j++)
        {            
            if (fscanf(contourVerticesFile, "%d %d", &xBuffer, &yBuffer) != 2)
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
        }
    }
    
    // close the input file
    fclose(contourVerticesFile);    
    
    return 0;
}

/* Function: getAllFilenamesInDirectory
 * 
 * Description: get all filenames in a given directory
 * 
 * Parameters:
 *     directoryName: name of the directory
 *     numFilesInDirectory: output int for number of files in directory
 *     filenamesInDirectory: output string array to store all filenames in given
 *         directory
 */
void getAllFilenamesInDirectory(
    _In_ char *directoryName,
    _Out_ int *numFilesInDirectory,
    _Out_ char **filenamesInDirectory)
{
    DIR *dir = opendir(directoryName);
    struct dirent *ent;
    
    *numFilesInDirectory = 0;
    
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) 
    {
        strcpy(filenamesInDirectory[(*numFilesInDirectory)], ent->d_name);
        (*numFilesInDirectory)++;
        
    }
    
    closedir (dir);
}

/* Function: getIndexOfMatchingString
 * 
 * Description: given a string and an array of strings, find the first string in
 *     the array that contains the input string as a substring
 * 
 * Parameters:
 *     strings: array of strings to search for substring in
 *     numStrings: number of strings in array
 *     stringToMatch: the substring to find in the array of strings
 * 
 * Returns: the first string in array that contains the given substring. If no
 *     string in array contains the substring, return NULL.
 */
int getIndexOfMatchingString(
    _In_ char **strings,
    _In_ int numStrings, 
    _In_ char *stringToMatch)
{
    for (int i = 0; i < numStrings; i++)
    {
        if (!strcmp(stringToMatch, strings[i]))
        {
            return i;
        }
    }
    
    return -1;
}

/* Function: read3DFeaturesFromFileForMatchingContours
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
 *     features3DFilename: 3D features filename
 *     features3D: CvPoint3D32f array to store the valid points for each contour
 *     numFeaturesInContours: array of ints containing number of valid features 
 *         for each contour
 *     numContours: number of contours
 *     contourNames: string array containing contour names
 * 
 * Returns: 0 on success, error code on error.
 */
int read3DFeaturesFromFileForMatchingContours(
    _In_ char *features3DFilename, 
    _InOut_ CvPoint3D32f **features3D,
    _InOut_ int *numFeaturesInContours,
    _In_ int numContours, 
    _In_ char **contourNames)
{
    // open the file for reading
    FILE *featuresFile = fopen(features3DFilename, "r");
    
    if (featuresFile == NULL)
    {
        return INPUT_FILE_OPEN_ERROR;
    }
    
    int numContourBuffer;
    
    // get the number of contours, including invalid features
    if (fscanf(featuresFile, "%d", &numContourBuffer) != 1)
    {
        return INCORRECT_INPUT_FILE_FORMAT_ERROR;
    }
    
    if (numContourBuffer < 1)
    {
        return INVALID_NUM_CONTOURS_ERROR;
    }
    
    char flagBuffer[1];
    float xBuffer, yBuffer, zBuffer;
    
    // for each contour in file, add the 3D features to features3D if the contour
    // name is in the contourNames array
    for (int i = 0; i < numContourBuffer; i++)
    {
        char contourNameBuffer[MAX_CONTOUR_NAME_LENGTH];
        int numFeaturesInContoursBuffer;
        
        if (fscanf(featuresFile, "%s %d", contourNameBuffer, &numFeaturesInContoursBuffer) != 2)
        {
            return INCORRECT_INPUT_FILE_FORMAT_ERROR;
        }
        
        // check to see if the contour name read in matches a name in the contourNames
        // array, if so, get the index
        int contourIndex = getIndexOfMatchingString(contourNames, numContours, contourNameBuffer);
        
        if (contourIndex == -1)
        {
            continue;
        }
                
        // set the numValidFeatures to the current number of contours for the
        // contour name
        int numValidFeatures = numFeaturesInContours[contourIndex];
        
        // for each feature in file, add to features3D array if valid
        for (int j = 0; j < numFeaturesInContoursBuffer; j++)
        {
            if (fscanf(featuresFile, "%s %f %f %f", flagBuffer, &xBuffer, &yBuffer, &zBuffer) != 4)
            {
                return INCORRECT_INPUT_FILE_FORMAT_ERROR;
            }
            
            if (flagBuffer[0] == '1')
            {                 
                features3D[contourIndex][numValidFeatures].x = xBuffer;
                features3D[contourIndex][numValidFeatures].y = yBuffer;
                features3D[contourIndex][numValidFeatures].z = zBuffer;            
                
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
        
        // update numValidFeatures for the contour
        numFeaturesInContours[contourIndex] = numValidFeatures;
    }

    // close the file
    fclose(featuresFile);
    
    return 0;
}

/* Function: main
 * 
 * Description: Main function to extract frames from 2 video files and runs the
 *     rest of the program using them. Takes at least 10 commandline arguments, 
 *     in the order: 
 *        <number of camera pairs>
 *        <pair 1 camera 1 filename>
 *        <pair 1 camera 1 frame number>
 *        <pair 1 camera 2 filename>
 *        <pair 1 camera 2 frame number>
 *        <pair 1 view name>
 *        <pair 1 camera coefficients filename>
 *        ...
 *        <TPS smoothing parameter>
 *        <feature detector>
 *        <output directory>
 * 
 * Parameters:
 *     argc: number of commandline arguments
 *     argv: string array of commandline arguments
 * 
 * Returns: 0 on success, 1 on error.
 */
int main (int argc, char *argv[])
{    
    // check for minimum number of commandline arguments
    if (argc < 11)
    {
        printf("Usage:\nvideos\n\t<number of camera pairs>\n\t<pair 1 camera 1 filename>\n\t<pair 1 camera 1 frame number>\n\t<pair 1 camera 2 filename>\n\t<pair 1 camera 2 frame number>\n\t<pair 1 view name>\n\t<pair 1 camera coefficients filename>\n\t...\n\t<TPS smoothing parameter>\n\t<feature detector>\n\t<output directory>\n");
        exit(1);
    }
    
    // get the number of camera pairs
    int numCameraPairs = atoi(argv[1]);
    
    if (numCameraPairs <= 0)
    {
        printf("Invalid number of camera pairs.\n");
        exit(1);
    }
    
    // number of commandline arguments should be numCameraPairs*6 + 5
    if (argc != numCameraPairs*6 + 5)
    {
        printf("Usage:\nvideos\n\t<number of camera pairs>\n\t<pair 1 camera 1 filename>\n\t<pair 1 camera 1 frame number>\n\t<pair 1 camera 2 filename>\n\t<pair 1 camera 2 frame number>\n\t<pair 1 view name>\n\t<pair 1 camera coefficients filename>\n\t...\n\t<TPS smoothing parameter>\n\t<feature detector>\n\t<output directory>\n");
        exit(1);
    }
    
    // allocate memory to store information for camera pairs
    char **camera1Filenames = (char **)malloc(numCameraPairs * sizeof(char *));
    int *camera1Frames = (int *)malloc(numCameraPairs * sizeof(int));
    
    if (camera1Filenames == NULL || camera1Frames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    char **camera2Filenames = (char **)malloc(numCameraPairs * sizeof(char *));
    int *camera2Frames = (int *)malloc(numCameraPairs * sizeof(int));
    
    if (camera2Filenames == NULL || camera2Frames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    char **cameraNames = (char **)malloc(numCameraPairs * sizeof(char *));
    
    if (cameraNames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    char **cameraCoefficientsFilenames = (char **)malloc(numCameraPairs * sizeof(char *));
    
    if (cameraCoefficientsFilenames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    int argIndex = 2;
    
    for (int i = 0; i < numCameraPairs; i++)
    {        
        camera1Filenames[i] = argv[argIndex];    
        camera1Frames[i] = atoi(argv[argIndex+1]);
        camera2Filenames[i] = argv[argIndex+2];
        camera2Frames[i] = atoi(argv[argIndex+3]);
        cameraNames[i] = argv[argIndex+4];
        cameraCoefficientsFilenames[i] = argv[argIndex+5];
        
        // make sure input video frames are valid
        if (camera1Frames[i] <= 0)
        {
            printf("Invalid frame number for pair %d camera 1.\n", i+1);
            exit(1);
        }
        
        if (camera2Frames[i] <= 0)
        {
            printf("Invalid frame number for pair %d camera 1.\n", i+1);
            exit(1);
        }
        
        // make sure input filenames are valid
        if (!fileExists(camera1Filenames[i]))
        {
            printf("Could not open pair %d camera 1 video file.\n", i+1);
            exit(1);
        }
        
        if (!fileExists(camera2Filenames[i]))
        {
            printf("Could not open pair %d camera 2 video file.\n", i+1);
            exit(1);
        }
        
        if (!fileExists(cameraCoefficientsFilenames[i]))
        {
            printf("Could not open pair %d camera coefficients file.\n", i+1);
            exit(1);
        }
        
        argIndex += 6;
    }
    
    double regularization = atof(argv[argIndex]);
    char *featureDetector = argv[argIndex+1];
    char *outputDirectory = argv[argIndex+2];
            
    // make sure input feature dectector is recognized
    if (strcasecmp(featureDetector, FAST_FEATURE_DETECTOR) &&        
        strcasecmp(featureDetector, GFTT_FEATURE_DETECTOR) &&      
        strcasecmp(featureDetector, SURF_FEATURE_DETECTOR) &&
        strcasecmp(featureDetector, SIFT_FEATURE_DETECTOR) &&
        strcasecmp(featureDetector, SPEEDSIFT_FEATURE_DETECTOR))
    {
        printf("Feature Detector not recognized. Please select from the following:\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n",
               FAST_FEATURE_DETECTOR,
               GFTT_FEATURE_DETECTOR,
               SURF_FEATURE_DETECTOR,
               SIFT_FEATURE_DETECTOR,
               SPEEDSIFT_FEATURE_DETECTOR);
        
        exit(1);
    }
    
    // make sure regularization parameter for TPS is valid
    if (regularization <= 0.0 || regularization == HUGE_VAL)
    {
        printf("Invalid smoothing parameter value.\n");
        exit(1);
    }
    
    // if output directory doesn't end with '/' char, append '/' to the string.
    // this is so we can later append a filename to the directory when we want 
    // to write the file to that directory
    if (outputDirectory[strlen(outputDirectory)-1] != '/')
    {
        strcat(outputDirectory, "/");
    }
    
    DIR *dir = opendir(outputDirectory);
    
    // if output directory does not exist, create it with correct permissions
    if (dir == NULL)
    {
        printf("Output directory does not exist.\n");
        
        if (mkdir(outputDirectory, S_IRWXO | S_IRWXG | S_IRWXU))
        {
            printf("Could not create output directory.\n");
            exit(1);
        }
        else
        {
            printf("Created output directory.\n");
        }
    }
    else
    {
        closedir(dir);
    }    
    
    // string for the MATLAB commands
    char command[500]; 
    
    Engine *matlabEngine;
    
    // open MATLAB engine
    if (!(matlabEngine = engOpen("\0")))
    {
        printf("Can't start MATLAB engine\n");        
        exit(1);
    }
    
    // create MATLAB arrays to retrieve values from MATLAB workspace
    mxArray **c1ImageData = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    mxArray **c1ImageDimensions = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    mxArray **c1ImagePaddedWidths = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    
    if (c1ImageData == NULL || c1ImageDimensions == NULL || c1ImagePaddedWidths == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    mxArray **c2ImageData = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    mxArray **c2ImageDimensions = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    mxArray **c2ImagePaddedWidths = (mxArray **)malloc(numCameraPairs * sizeof(mxArray *));
    
    if (c2ImageData == NULL || c2ImageDimensions == NULL || c2ImagePaddedWidths == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    // create IplImage arrays for camera 1 and 2 images for all camera pairs
    IplImage **c1Images = (IplImage **)malloc(numCameraPairs * sizeof(IplImage *));
    IplImage **c2Images = (IplImage **)malloc(numCameraPairs * sizeof(IplImage *));
    
    if (c1Images == NULL || c2Images == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    // for each camera pair, get the specified frames from cameras 1 and 2, using
    // MATLAB functions
    for (int i = 0; i < numCameraPairs; i++)
    {
        char video1Extension[6];
        
        // get the video file extension for the first video file
        if (getVideoFileExtension(camera1Filenames[i], video1Extension) == INVALID_VIDEO_FILE_EXTENSION_ERROR)
        {
            printf("Video files must be of extension .mrf or .cine.\n");
            exit(1);
        }
        
        // call appropriate MATLAB function depending on whether video file is .cine 
        // or .mrf to extract the frame as a MATLAB image. If neither, error.
        if ((strcasecmp(video1Extension, ".cine") == 0) || (strcasecmp(video1Extension, ".cin") == 0))
        {
            sprintf(command, "c1 = cineRead('%s', %d);", camera1Filenames[i], camera1Frames[i]);
            engEvalString(matlabEngine, command);
        }
        else if (strcasecmp(video1Extension, ".mrf") == 0)
        {
            sprintf(command, "c1 = mrfRead('%s', %d);", camera1Filenames[i], camera1Frames[i]);
            engEvalString(matlabEngine, command);
        }
        else
        {
            printf("Videos must be of extension .mrf or .cine.\n");
            exit(1);
        }
        
        char video2Extension[6];
        
        // get the video file extension for the second video file
        if (getVideoFileExtension(camera2Filenames[i], video2Extension) == INVALID_VIDEO_FILE_EXTENSION_ERROR)
        {
            printf("Video files must be of extension .mrf or .cine.\n");
            exit(1);
        }
        
        // call appropriate MATLAB function depending on whether video file is .cine 
        // or .mrf to extract the frame as a MATLAB image. If neither, error.
        if ((strcasecmp(video2Extension, ".cine") == 0) || (strcasecmp(video2Extension, ".cin") == 0))
        {
            sprintf(command, "c2 = cineRead('%s', %d);", camera2Filenames[i], camera2Frames[i]);
            engEvalString(matlabEngine, command);
        }
        else if (strcasecmp(video2Extension, ".mrf") == 0)
        {
            sprintf(command, "c2 = mrfRead('%s', %d);", camera2Filenames[i], camera2Frames[i]);
            engEvalString(matlabEngine, command);
        }
        else
        {
            printf("Videos must be of extension .mrf or .cine.\n");
            exit(1);
        }
        
        // call MATLAB function convert_image_matlab2cv_gs on MATLAB images to convert
        // them into a format that will be compatible with the IplImages of OpenCV
        sprintf(command, "[c1_img c1_dim c1_padded_width] = convert_image_matlab2cv_gs(c1);");    
        engEvalString(matlabEngine, command);
        
        sprintf(command, "[c2_img c2_dim c2_padded_width] = convert_image_matlab2cv_gs(c2);");
        engEvalString(matlabEngine, command);
        
        // retrieve the image data, image dimensions, and image padded width variables 
        // from MATLAB for both camera images
        c1ImageData[i] = engGetVariable(matlabEngine, "c1_img");
        c1ImageDimensions[i] = engGetVariable(matlabEngine, "c1_dim");
        c1ImagePaddedWidths[i] = engGetVariable(matlabEngine, "c1_padded_width");
        
        c2ImageData[i] = engGetVariable(matlabEngine, "c2_img");
        c2ImageDimensions[i] = engGetVariable(matlabEngine, "c2_dim");
        c2ImagePaddedWidths[i] = engGetVariable(matlabEngine, "c2_padded_width");    
        
        if (c1ImageData[i] == NULL || 
            c1ImageDimensions[i] == NULL || 
            c1ImagePaddedWidths[i] == NULL)
        {        
            printf("Could not retrieve all necessary information for pair %d camera 1 frame %d from MATLAB.\n", i+1, camera1Frames[i]);
            exit(1);
        }
        
        if (c2ImageData[i] == NULL || 
            c2ImageDimensions[i] == NULL || 
            c2ImagePaddedWidths[i] == NULL)
        {        
            printf("Could not retrieve all necessary information for pair %d camera 2 frame %d from MATLAB.\n", i+1, camera2Frames[i]);
            exit(1);
        }
        
        int c1Status, c2Status;
        
        ImageInfo c1ImageInfo, c2ImageInfo;            
        
        // extract the image information from the MATLAB variables in the form of 
        // mxArrays, and store in ImageInfo structs
        c1Status = getInputImageInfo(&c1ImageInfo, c1ImageData[i], c1ImageDimensions[i], c1ImagePaddedWidths[i]);
        c2Status = getInputImageInfo(&c2ImageInfo, c2ImageData[i], c2ImageDimensions[i], c2ImagePaddedWidths[i]);
        
        if (c1Status == IMAGE_INFO_DATA_ERROR)
        {
            printf("Pair %d camera 1: Images must have two dimensions.\n", i+1);
            exit(1);
        }
        
        if (c2Status == IMAGE_INFO_DATA_ERROR)
        {
            printf("Pair %d camera 2: Images must have two dimensions.\n", i+1);
            exit(1);
        }
        
        if (c1Status == IMAGE_INFO_DIMENSIONS_ERROR)
        {
            printf("Pair %d camera 1: Image dimension vectors must contain two elements: [width, height].\n", i+1);
            exit(1);
        }
        
        if (c2Status == IMAGE_INFO_DIMENSIONS_ERROR)
        {
            printf("Pair %d camera 2: Image dimension vectors must contain two elements: [width, height].\n", i+1);
            exit(1);
        }
        
        if (c1Status == IMAGE_INFO_PADDED_WIDTH_ERROR)
        {
            printf("Pair %d camera 1: Padded image widths must be scalars.\n", i+1);
            exit(1);
        }
        
        if (c2Status == IMAGE_INFO_PADDED_WIDTH_ERROR)
        {
            printf("Pair %d camera 2: Padded image widths must be scalars.\n", i+1);
            exit(1);
        }
        
        if (c1Status == IMAGE_DEPTH_ERROR)
        {
            printf("Pair %d camera 1: Images must be represented by 8 or 16-bit integers.\n", i+1);
            exit(1);
        }
        
        if (c2Status == IMAGE_DEPTH_ERROR)
        {
            printf("Pair %d camera 2: Images must be represented by 8 or 16-bit integers.\n", i+1);
            exit(1);
        }
        
        // create IplImages using values in ImageInfo structs
        c1Status = createIplImageFromImageInfo(&(c1Images[i]), c1ImageInfo);
        c2Status = createIplImageFromImageInfo(&(c2Images[i]), c2ImageInfo);
        
        if (c1Status == OUT_OF_MEMORY_ERROR ||
            c2Status == OUT_OF_MEMORY_ERROR)
        {
            printf("Out of memory error.\n");
            exit(1);
        }
        
        // flip the images over the y-axis to compensate for the differences in axial
        // labels between MATLAB and OpenCV (camera coefficients would not correctly
        // correspond to image otherwise)
        cvFlip(c1Images[i], NULL, 1);
        cvFlip(c2Images[i], NULL, 1);
    }
    
    char errorMessage[500];
    
    int numContours;
    char **contourNames;
    CvPoint3D32f **features3D;
    char **validFeatureIndicator;
    int *numFeaturesInContours;
    
    char contoursFilename[MAX_FILENAME_LENGTH];
    
    // for each camera pair, run features and triangulation
    for (int i = 0; i < numCameraPairs; i++)
    {
        // create the output 2D features filename as "frame<frame number>_features2D_<camera name>.txt"
        char features2DFilename[MAX_FILENAME_LENGTH];    
        sprintf(features2DFilename, "%sframe%d_features2D_%s.txt", outputDirectory, camera1Frames[i], cameraNames[i]);
        
        // create the output contours filename as "frame<frame number>_contours_<camera name>.txt"
        char tempContoursFilename[MAX_FILENAME_LENGTH];    
        sprintf(tempContoursFilename, "%sframe%d_contours_%s.txt", outputDirectory, camera1Frames[i], cameraNames[i]);
        
        printf("Camera pair for %s view:\n", cameraNames[i]);
        
        // run the features program to extract matching 2D features from the 2 
        // images within user defined contour
        if (features(c1Images[i], c2Images[i], features2DFilename, tempContoursFilename, featureDetector, errorMessage))
        {
            printf("Features: %s\n", errorMessage);
            exit(1);
        }
        
        // we only need to save the contour(s) for the first camera pair, as that 
        // is the one we will use to create the meshes, and we only use the contours
        // with the same name(s) in subsequent camera pairs
        if (i == 0)
        {
            strcpy(contoursFilename, tempContoursFilename);
            
            // get the contour names of the contours selected in features function for
            // output file naming and contour matching in other camera pairs
            int status = readContourNamesFromInputFile(&numContours, &contourNames, contoursFilename);
            
            if (status == INPUT_FILE_OPEN_ERROR)
            {
                printf("Could not open contour vertices file.\n");
                exit(1);
            }
            
            if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
            {
                printf("Contour vertices file has incorrect format.\n");
                exit(1);
            }
            
            if (status == OUT_OF_MEMORY_ERROR)
            {
                printf("Out of memory error.\n");
                exit(1);
            }
            
            // allocate memory for 3D features
            features3D = (CvPoint3D32f **)malloc(numContours * sizeof(CvPoint3D32f *));
            validFeatureIndicator = (char **)malloc(numContours * sizeof(char *));
            numFeaturesInContours = (int *)malloc(numContours * sizeof(int));
            
            if (features3D == NULL || numFeaturesInContours == NULL || validFeatureIndicator == NULL)
            {
                printf("Out of memory error.\n");
                exit(1);
            }
            
            for (int j = 0; j < numContours; j++)
            {
                features3D[j] = (CvPoint3D32f *)malloc(MAX_FEATURES_IN_CONTOUR * sizeof(CvPoint3D32f));
                validFeatureIndicator[j] = (char *)malloc(MAX_FEATURES_IN_CONTOUR * sizeof(char));
                
                if (features3D[j] == NULL || validFeatureIndicator[j] == NULL)
                {
                    printf("Out of memory error.\n");
                    exit(1);
                }
                
                numFeaturesInContours[j] = 0;
            }
        }
        
        // create the output 3D features filename as "frame<frame number>_features3D_<camera name>.txt"
        char features3DFilename[MAX_FILENAME_LENGTH];    
        sprintf(features3DFilename, "%sframe%d_features3D_%s.txt", outputDirectory, camera1Frames[i], cameraNames[i]);
        
        // triangulate the matching 2D features between cameras to find the 3D coordinates 
        // of the features, and remove invalid features
        if (triangulation(cameraCoefficientsFilenames[i], features2DFilename, features3DFilename, c1Images[i], errorMessage))
        {
            printf("Triangulation: %s\n", errorMessage);
            exit(1);
        }
        
        // if features from triangulation lie within contours that have the same
        // names as those defined for the first camera pair, add them to the
        // 3D features array for mesh creation
        int status = read3DFeaturesFromFileForMatchingContours(features3DFilename, features3D, numFeaturesInContours, numContours, contourNames);
        
        if (status == INPUT_FILE_OPEN_ERROR)
        {
            printf("Could not open 3D features file.\n");
            exit(1);
        }
        
        if (status == INVALID_NUM_CONTOURS_ERROR)
        {
            printf("At least 1 contour region required.\n");
            exit(1);
        }
        
        if (status == INCORRECT_INPUT_FILE_FORMAT_ERROR)
        {
            printf("3D features file has incorrect format.\n");
            exit(1);
        }
    }        
    
    // for each contour (defined for the first camera pair), perform RANSAC on
    // the cumulative 3D features from all camera pairs that lie within the contour
    for (int i = 0; i < numContours; i++)
    {    
        memset(validFeatureIndicator[i], 1, numFeaturesInContours[i] * sizeof(char));

        // perform RANSAC to remove points that lie too far off a best-fit surface
        if (ransac(features3D[i], validFeatureIndicator[i], numFeaturesInContours[i], errorMessage))
        {
            printf("RANSAC: %s\n", errorMessage);
            exit(1);
        }
        
        int numValidFeatures = 0;
        
        for (int j = 0; j < numFeaturesInContours[i]; j++)
        {
            if (validFeatureIndicator[i][j])
            {
                numValidFeatures++;
            }
        }
        
        printf("Total valid features after RANSAC for contour %s: %d\n", contourNames[i], numValidFeatures);

    }
    
    // create the output 3D features filename for all camera pairs as 
    // "frame<frame number>_features3D.txt", and write the result of RANSAC to
    // the file
    char features3DFilename[MAX_FILENAME_LENGTH];    
    sprintf(features3DFilename, "%sframe%d_features3D.txt", outputDirectory, camera1Frames[0]);
    
    int status = write3DFeaturesToFile(features3D, validFeatureIndicator, numFeaturesInContours, contourNames, numContours, features3DFilename);
    
    if (status == OUTPUT_FILE_OPEN_ERROR)
    {
        sprintf(errorMessage, "Could not open output file.");
        return 1;
    }
    
    char **meshFilenames = (char **)malloc(numContours * sizeof(char *));
    
    if (meshFilenames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    // for each contour, create a different mesh output file
    for (int i = 0; i < numContours; i++)
    {
        meshFilenames[i] = (char *)malloc(MAX_FILENAME_LENGTH * sizeof(char));
        
        if (meshFilenames[i] == NULL)
        {
            printf("Out of memory error.\n");
            exit(1);
        }
        
        // create the output mesh filename as "frame<frame number>_mesh_<contour name>_<camera name>.txt"
        sprintf(meshFilenames[i], "%sframe%d_mesh_%s.txt", outputDirectory, camera1Frames[0], contourNames[i]);
    }
    
    // create the wing meshes from the triangulated 3D points and the user-selected
    // contours, and write each mesh to a different file for each contour
    if (mesh(features3DFilename, contoursFilename, cameraCoefficientsFilenames[0], meshFilenames, numContours, regularization, errorMessage))
    {
        printf("Mesh: %s\n", errorMessage);
        exit(1);
    }
    
    // we only calculate the flow of a wing mesh if there is a mesh file with the
    // same contour name in the output directory for the previous video frame
    char **flowFilenames = (char **)malloc(numContours * sizeof(char *));
    
    if (flowFilenames == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    for (int i = 0; i < numContours; i++)
    {
        flowFilenames[i] = NULL;
    }
    
    int numFilesInDirectory;
    char **filenamesInDirectory = (char **)malloc(MAX_FILES_IN_DIRECTORY * sizeof(char *));
    
    if (filenamesInDirectory == NULL)
    {
        printf("Out of memory error.\n");
        exit(1);
    }
    
    for (int i = 0; i < MAX_FILES_IN_DIRECTORY; i++)
    {
        filenamesInDirectory[i] = (char *)malloc(MAX_FILENAME_LENGTH * sizeof(char));
        
        if (filenamesInDirectory[i] == NULL)
        {
            printf("Out of memory error.\n");
            exit(1);
        }
    }
    
    // get all files in the output directory
    getAllFilenamesInDirectory(outputDirectory, &numFilesInDirectory, filenamesInDirectory);
     
    // for each contour check if previous frame mesh file for same contour exists
    // in output directory
    for (int i = 0; i < numContours; i++)
    {
        // set substring indicating match to be "frame<previous frame number>_mesh_<contour name>.txt"
        char filenameToMatch[MAX_FILENAME_LENGTH];
        sprintf(filenameToMatch, "frame%d_mesh_%s.txt", camera1Frames[0]-1, contourNames[i]);
        
        // try to find a filename from the output directory that contains the
        // substring indicating a match for a previous frame mesh for the same
        // contour
        int fileExists = getIndexOfMatchingString(filenamesInDirectory, numFilesInDirectory, filenameToMatch);
        
        // if filename was found, create a flow output file for current contour 
        // and call flow to calculate the flow between previous contour mesh and 
        // current contour mesh
        if (fileExists != -1)
        {
            flowFilenames[i] = (char *)malloc(MAX_FILENAME_LENGTH * sizeof(char));
            
            if (flowFilenames[i] == NULL)
            {
                printf("Out of memory error.\n");
                exit(1);
            }
            
            // create the output flow filename as "frame<frame number>_flow_<contour name>_<camera name>.txt"
            sprintf(flowFilenames[i], "%sframe%d_flow_%s.txt", outputDirectory, camera1Frames[0], contourNames[i]);
            
            // add the output directory name to the beginning of the previous mesh
            // filename
            char prevFrameMeshFile[MAX_FILENAME_LENGTH];
            sprintf(prevFrameMeshFile, "%s%s", outputDirectory, filenameToMatch);
            
            // call flow to find the flow between the previous mesh file and the
            // current mesh file for each mesh point current contour
            if (flow(prevFrameMeshFile, meshFilenames[i], flowFilenames[i], errorMessage))
            {
                printf("Flow: %s\n", errorMessage);
                exit(1);
            }
        }
        
        else
        {
            printf("Mesh points file for previous frame not found for contour %s. Unable to calculate flow.\n", contourNames[i]);
        }
    }
    
    sprintf(command, "hold on;");
    engEvalString(matlabEngine, command);
    
    // for each contour, display MATLAB 3D plot of the mesh, as well as the flow 
    // for the mesh, if applicable
    for (int i = 0; i < numContours; i++)
    {        
        if (flowFilenames[i] != NULL)
        {
            sprintf(command, "flows = load('%s');", flowFilenames[i]);
            engEvalString(matlabEngine, command);
            
            // plot the flows of the mesh points
            sprintf(command, "quiver3(flows(:,1), flows(:,2), flows(:,3), flows(:,4), flows(:,5), flows(:,6), 4, 'r-');");
            engEvalString(matlabEngine, command);
            
        }
        
        sprintf(command, "mesh = importdata('%s', ' ', 1);", meshFilenames[i]);
        engEvalString(matlabEngine, command);
        
        // plot the mesh points
        sprintf(command, "plot3(mesh.data(:,1), mesh.data(:,2), mesh.data(:,3), 'b.');");
        engEvalString(matlabEngine, command);
    }
    
    // reverse the z and y coordinates in the display
    sprintf(command, "set(gca,'zdir','reverse','ydir','reverse');");
    engEvalString(matlabEngine, command);
    
    // scale the axes to be equal
    sprintf(command, "axis equal");
    engEvalString(matlabEngine, command);
    
    // wait for the user to hit enter
    printf("Hit return to continue.\n");
    fgetc(stdin);
    
    // close MATLAB engine
    engClose(matlabEngine);
    
    // cleanup
    free(camera1Filenames);
    free(camera1Frames);
    free(camera2Filenames);
    free(camera2Frames);
    free(cameraNames);
    free(cameraCoefficientsFilenames);
    
    for (int i = 0; i < numCameraPairs; i++)
    {
        mxDestroyArray(c1ImageData[i]);
        mxDestroyArray(c1ImageDimensions[i]);
        mxDestroyArray(c1ImagePaddedWidths[i]);
        
        mxDestroyArray(c2ImageData[i]);
        mxDestroyArray(c2ImageDimensions[i]);
        mxDestroyArray(c2ImagePaddedWidths[i]);
        
        free(c1Images[i]->imageData);
        cvReleaseImageHeader(&c1Images[i]);
        
        free(c2Images[i]->imageData);
        cvReleaseImageHeader(&c2Images[i]);
    }
    
    free(c1ImageData);
    free(c1ImageDimensions);
    free(c1ImagePaddedWidths);
    
    free(c2ImageData);
    free(c2ImageDimensions);
    free(c2ImagePaddedWidths);
    
    free(c1Images);
    free(c2Images);
    
    for (int i = 0; i < MAX_FILES_IN_DIRECTORY; i++)
    {
        free(filenamesInDirectory[i]);
    }
    
    free(filenamesInDirectory);
    
    for (int i = 0; i < numContours; i++)
    {
        free(contourNames[i]);
        free(features3D[i]);
        free(validFeatureIndicator[i]);
                
        free(meshFilenames[i]);
        
        if (flowFilenames[i] != NULL)
        {
            free(flowFilenames[i]);
        }
    }
    
    free(contourNames);
    free(features3D);
    free(validFeatureIndicator);
    free(numFeaturesInContours);
    
    free(meshFilenames);
    free(flowFilenames);
    
    exit(0);
}
