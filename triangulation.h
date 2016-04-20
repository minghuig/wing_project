#ifndef __TRIANGULATION_H
#define __TRIANGULATION_H

int triangulation(_In_ char *cameraCoefficientsFilename, _In_ char *featuresFilename, _In_ char *features3DFilename, _In_ IplImage *displayImage, _Out_ char *errorMessage);
int readCoefficientsFromInputFile(_Out_ double ***cameraCoefficients, _Out_ int *numCameras, _In_ char *filename);
int write3DFeaturesToFile(_In_ CvPoint3D32f **features3D, _In_ char **validFeatureIndicator, _In_ int *numFeaturesInContours, _In_ char **contourNames, _In_ int numContours, _In_ char *outputFilename);

#endif