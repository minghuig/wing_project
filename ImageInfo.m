% Class: ImageInfo
% 
% Description: A class that contains information on an image for passing 
%     into C for OpenCV use.
%
% Properties:
%     image_data: The image data.
%     image_dimensions: The dimensions of the image, stored as a 2x1 vector
%         containing [width, height].
%     image_padded_width: The width of the image including the width of the
%         zero-padding used by OpenCV images
%
% Author: Ming Guo
% Created: 6/30/11

classdef ImageInfo
    properties
        image_data;
        image_dimensions;
        image_padded_width;
    end
end