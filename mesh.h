#ifndef __MESH_H
#define __MESH_H

int mesh(_In_ char *features3DFilename, _In_ char *contoursFilename, _In_ char *cameraCoefficientsFilename, _In_ char **meshPointsFilenames, _In_ int numMeshFiles, _In_ double regularization, _Out_ char *errorMessage);

#endif