% Function: convert_image_matlab2cv_gs
% 
% Description: Function to convert an image from format used by MATLAB to 
%     format used by OpenCV.
%
%     Coverts to grayscale if input image is RGB. 
%
%     Converts to 8-bit unsigned integer representation if image uses 
%     anything other than 8 or 16-bit unsigned integers
% 
%     Differences between grayscale MATLAB images and OpenCV images:
%       - Matlab stores images as height × width, while OpenCV stores 
%         images as width × height
%       - For performance reasons, OpenCV pads the images with zeros so 
%         that the number of columns in an image is divisible by 4 (except 
%         for BMPs, which this function does not handle).
%
% Parameters:
%     ml_img: MATLAB-formatted image to be converted.
%
% Returns:
%     cv_img_info: An ImageInfo object containing the data, dimensions, and
%         padded width of the input image.
%
% Author: Ming Guo
% Created: 6/21/11

function [cv_img dim padded_width] = convert_image_matlab2cv_gs(ml_img)

% set Gamma gain (values less than 1 brighten the image)
gamma=0.5;

% if image is RGB, convert to grayscale
if (ndims(ml_img) == 3)
    ml_img = rgb2gray(ml_img);
end

maxVal = max(max(ml_img));

ml_img = double(ml_img).^gamma;

if (maxVal > intmax('uint8'))
    ml_img = uint16((ml_img)/(max(max(ml_img)))*double(intmax('uint16')));
else
    ml_img = uint8((ml_img)/(max(max(ml_img)))*double(intmax('uint8')));
end

% get image dimensions
width = size(ml_img, 2);
height = size(ml_img, 1);

dim = [width, height];

% get padding amount
padding_amount = 4 - mod(width, 4);

% pad image if width is not divisible by 4
if (padding_amount < 4)
    padded_width = width + padding_amount;
    
    if (strcmpi(class(ml_img), 'uint16'))
        padding_matrix = zeros(height, padding_amount, 'uint16');
    else
        padding_matrix = zeros(height, padding_amount, 'uint8');
    end
    
    ml_img = [ml_img padding_matrix];
else
    padded_width = width;
end

% exchange rows and columns
cv_img = permute(ml_img(:, end:-1:1), [2 1]);

% cv_img_info = ImageInfo();
% cv_img_info.image_data = cv_img;
% cv_img_info.image_dimensions = dim;
% cv_img_info.image_padded_width = padded_width;
