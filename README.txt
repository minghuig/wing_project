to compile program on lab computer: mex -f /usr/local/MATLAB/R2011a/bin/engopts.sh wing.cpp features.cpp mesh.cpp flow.cpp triangulation.cpp ransac_wing.cpp /usr/lib/libgsl.so -lgslcblas /usr/local/lib/libopencv_objdetect.so.2.3 /usr/local/lib/libopencv_core.so.2.3 /usr/local/lib/libopencv_highgui.so.2.3 /usr/local/lib/libopencv_features2d.so.2.3 /usr/local/lib/libopencv_video.so.2.3
  - specify location of engopts.sh, which is the shell script for configuring MATLAB engine standalone applications
  - include all relevant cpp files (wing.cpp, features.cpp, mesh.cpp, flow.cpp, triangulation.cpp, ransac_wing.cpp) in the compile command
  - specify location of libgsl.so, required for gsl functions called in program
  - include "-lgslcblas" to specify usage of the gsl C interface to the Basic Linear Algebra Subprograms
  - specify locations of libopencv_objdetect.so.2.3, libopencv_core.so.2.3, libopencv_highgui.so.2.3, libopencv_features2d.so.2.3, libopencv_video.so.2.3, as they are the OpenCV libraries required by the program
  - "cineRead.m", "cineInfo.m", "mrfRead.m", "mrfInfo.m", "ImageInfo.m", and "convert_image_matlab2cv_gs.m" are MATLAB script files that are called by the program, so the program needs to be able to find them.

to run program: wing 
                    <number of camera pairs> 
                    <pair 1 camera 1 filename> 
                    <pair 1 camera 1 frame number> 
                    <pair 1 camera 2 filename> 
                    <pair 1 camera 2 frame number> 
                    <pair 1 view name> 
                    <pair 1 camera coefficients filename> 
                    ...
                    <TPS smoothing value> 
                    <feature detector> 
                    <output directory name>
  - only 1 camera pair is needed for high-resolution videos where the wing is in good focus and there is a good view of the entire surface. otherwise, specifying multiple camera pairs may give a better mesh, although it is more time-consuming for the user to select from additional camera views.
  - the contour(s) selected for the first camera pair will be the only one used for mesh creation, so it is highly recommended to enter the parameters for the camera pair with the best view of the wing surface as the first pair
  - video files must be .mrf or .cin/.cine
  - video frame numbers must be greater than 0. If a frame number is greater than the number of frames in the video, an error will be thrown from MATLAB.
  - camera coefficient files contains the coefficients for n cameras (n must be >= 2, and only the first 2 cameras are used by the program as it is now) and must be in the format:
      <number of cameras>
      camera_1
      <1st coefficient for camera 1>
      ...
      <11th coefficient for camera 1>
      ...
      camera_n
      <1st coefficient for camera n>
      ...
      <11th coefficient for camera n>
  - camera view parameters are used for output file naming purposes and can be arbitrary (but should be meaningful)
  - TPS smoothing value can be any float greater than 0. Would recommend a value less than 1 for surfaces that have a natural curve.
  - feature detector supported are FAST, GFTT, SURF, SIFT, and SPEEDSIFT
    - FAST and GFTT use optic flow tracking to find matches between cameras 1 and 2. Faster than SURF/SIFT, but only works well for high-resolution, in-focus images
    - SURF and SIFT use feature matching to find matches between cameras 1 and 2. SURF is much faster than SIFT, but SIFT works better for lower quality images.
    - SPEEDSIFT uses the SIFT algorithm to find and match features between cameras 1 and 2. It prompts the user to select via bounding box the general area of the wing from the camera 2 image (additional user input). It then blacks out the background for both the camera 1 image (anything not within the selected contours) and the camera 2 image (anything outside the selected bounding box). This will limit the number of features found for both images and thus speed up SIFT considerably.
  - if running program on consecutive frames of the same video (doesn't matter which camera view), it is recommended to use the same output directory, so that velocity flow will be computed for the meshes between consecutive frames for matching contours.

user input during program:
  - for each camera pair:
    - in first image that pops up, drag a bounding box around the area of interest (where the wing is) to magnify the region and make wing contour selection easier. Use left mouse button to drag and right mouse button to reset. Hit enter when done.
    - in second image that pops up (magnified region that was previous selected), click with the left mouse button around the contours of the wing(s). Click with the right mouse button to delete the previous contour vertex. When a contour is finished, hit spacebar. Multiple contours can be selected. If you right click after a contour is finished and before you start the next contour, you will delete the entire previous finished contour. Hit enter when done.
    - next, the commandline will prompt user to enter names for each contour selected. 
      - Names can be arbitrary but should be meaningful. 
      - This feature is not particular robust. If you enter a contour name longer than 100 characters, you will overwrite the buffer, so don't unless you're trying to hack the program. 
      - *IMPORTANT* For consistency, you must enter the names in the order in which you selected their respective contours. For example, if you selected the contour for the left wing first, then the contour for the right wing, you must enter the name for the left wing contour first, then the name for the right wing contour. 
      - *IMPORTANT* If you're using more than one camera pair, for camera pairs 2 through n, only contours with names matching those of the first camera pair will be used in mesh creation, so name accordingly.
    - if SPEEDSIFT is selected as the feature detector, an additional user-input image will pop up. Here, use the left mouse button to drag a bounding box around a region that encompasses all contours selected in the previous image. Make the bounding box as tight as possible for maximum program speed-up. Again, right click to reset and hit enter when done.
    - another image will pop up depicting valid matched features and their 2D re-projections after 3D triangulation. This was for debugging purposes, and you can comment it out from the code if you'd like (in triangulation.cpp).
  - MATLAB plot will then pop up showing the end resulting mesh, as well as the flow, if the mesh for a previous frame for the same contour name was found in the output directory
  - if error at any point, program will exit and (hopefully) print out an appropriate error message

intermediary output:
  - the program will output the following files (the format of which can be found in the documentation of functions in the code that writes and reads the files):
    - "frame<frame number>_contours_<camera name>.txt" -- contains the user-selected contour information for particular camera view
    - "frame<frame number>_features2D_<camera name>.txt" -- contains the 2D features found and matched between cameras 1 and 2 for particular camera view
    - "frame<frame number>_features3D_<camera name>.txt" -- contains the triangulated 3D features, as well as a bit indicating whether the feature is found to be valid or not, for particular camera view
    - "frame<frame number>_features3D.txt" -- contains the cumulative valid 3D features for all camera views, with a bit indicating whether or not it was found valid by RANSAC
    - "frame<frame number>_mesh_<contour name>.txt" -- a different mesh file will be outputted for every contour selected by user. It contains all 3D points in the mesh.
    - "frame<frame number>_flow_<contour name>.txt" -- a different flow file will be outputted for every contour selected by user, if a mesh file for a previous frame for the same contour name exists in output directory. It contains all 3D points for the mesh and the delta x, delta y, and delta z of their flows from the previous from to the current frame.
  - for "frame<frame number>_contours_<camera name>.txt", "frame<frame number>_features2D_<camera name>.txt", and "frame<frame number>_features3D_<camera name>.txt", the frame number in the output file names will be the frame number for camera 1 of the camera pair
  - for "frame<frame number>_features3D.txt", "frame<frame number>_mesh_<contour name>.txt", and "frame<frame number>_flow_<contour name>.txt", the frame number in the output file names will be the frame number for camera 1 of camera pair 1

future directions (in order of importance/usefulness, in my opinion):
  - better user interface (graphical)
  - automatic wing edge detection (eliminate user input for wing contours, which is a time bottleneck) -- edge detection is available in OpenCV, but separating the wing edges from the body/background is a more difficult problem
  - multiple cameras views (instead of limiting to pair of 2 cameras)

changing program to model other types of 3D objects (besides relatively flat wing surfaces):
  - change the RANSAC code so that the model 3D shape is whatever 3D shape you want (it is currently a plane). getsModel and fitsModel are the 2 functions that need to be replaced, and the MIN_POINTS_NEEDED_FOR_MODEL and NUM_LINEAR_EQUATION_PARAMS_FOR_MODEL values may need to be changed.
  - replace the mesh code with code that will create a regular 3D grid of desired shape fitted to triangulated 3D features
