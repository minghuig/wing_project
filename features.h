#ifndef __FEATURES_H
#define __FEATURES_H

int features(_In_ IplImage *camera1Frame1, _In_ IplImage *camera2Frame1, _In_ char *featuresFilename, _In_ char *contoursFilename, _In_ char *featureDetector, _Out_ char *errorMessage);
void drawFlowArrows(_InOut_ IplImage *opticFlowImage, _In_ CvPoint p, _In_ CvPoint q, _In_ int minFlowLength, _In_ int lengtheningFactor, _In_ int arrowScale);
int areFeaturesInContour(_In_ CvPoint2D32f *features2D, _In_ int numFeatures, _Out_ char *validFeatureIndicator, _In_ CvSeq *contour);

#endif