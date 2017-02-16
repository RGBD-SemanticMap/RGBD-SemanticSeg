img = rgb2gray(imread('E:\lena.jpg'));
glcm = graycomatrix(img,'NumLevels',256);
glcm = 1.0*glcm/max(max(glcm));
glcm = sqrt(sqrt(glcm));
imshow(glcm, []);