cd /Users/ben/Dropbox/Projects/GMMMIL/data/  
img = imread('image.png');
img1 = img(1:300,1:300,1);
win1 = strel('arbitrary',ones(10));
win2 = strel('arbitrary',ones(20));
imgMorph1 = imdilate(imdilate(img1,win2),win2);
imgMorph2 = imdilate(imdilate(imgMorph1,win2),win2);
imgMorph3 = imdilate(imdilate(imgMorph2,win2),win2);
% imshow(imgErode)
% imshow(img1)
subplot(1,3,1), imshow(img1)
subplot(1,3,2), imshow(imgMorph2)
subplot(1,3,3), imshow(imgMorph3)