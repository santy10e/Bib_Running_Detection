This folder contains set #3 RBNR images:
 Images (jpg extension) 
 Reference ground truth files (mat extension)

The set includes 58 color images in jpg format
Image resolution: 768x1024
Image size: ~187-407KB

 For each image a corresponding reference file with the same name and with '.mat' extension. Each reference file contains one or more RBNs (number + location) in the following format:

Face location:
 facep(i,:) = [minRow maxRow minCol maxCol] 

RBN location:
 tagp(i,:)  = [minRow maxRow minCol maxCol]

RBN (number)
 number(i)  = [number]

 where
 i=1:N the number of RBNs in the image,
minRow = the upper row of the bounding box (row with the lowest index)
maxRow = the bottom row of the bounding box (row with the highest index)
minCol = the upper row of the bounding box (row with the lowest index)
maxCol = the bottom row of the bounding box (row with the highest index)

A summary of the ground truth RBNs of each image can be find in the "list.txt" file in this folder.

