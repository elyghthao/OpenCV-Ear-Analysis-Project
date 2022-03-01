# OpenCV-Ear-Analysis-Project


This project was done to get familiar with opencv libraries. It resulted in me becoming not only very familiar with opencv libaries, but also the computer vision strategies and logic to compute the nessecary components to identify human ears.
# Lessons Learned
Getting depth with stereo vision is a math heavy and error prone process. We first need to find the intrinsic parameters by calibrating cameras using chessboard images. Find the extrinsic parameters with key point matching to learn the relationship between the left image and right image. Then, we can project the images to a common plate to get the correct epipolar lines. Solve the correspondence problem, calculate disparity, and finally calculate the depth of the images. Having to handle each of these problems concerning stereo vision was very hard to do without using code samples. 


SIFT keypoint matching is heavily impacted by a lot of different factors, ranging from brightness, contrast, color, size of image, and blurs. When we first started we thought the most we would be changing would be size, but as you can see that quickly changed as the project progressed. 


Match Template is very picky on what it will successfully identify. Our handling of sizes and resizes, while did work to an extent, was in no way robust and would probably result in more issues if applied to images outside of our sample. 


When dealing with image processing, it is very important to have a standardized size for test images in terms of pixels and resolution. With different image sizes, it not only affected our performance in the code, but also how well our algorithms were used.


A lot of the lessons in the class were easily reviewed and reapplied in our learning throughout this project. From using opencv methods to create different types of images(edge detection, blurs), all the way to what we had learned recently in our lectures about stereo vision.

 # The Project
The basis behind this project was to create an image identification program that could identify ears without relying on any ai neural networking. Our process was long, but the end results reached a point that could easily be expanded on into a fully developed program fully tuned to identifying ears. 
  
Below is the documentation explaining my teams process and concluding opencv strategies we chose to use
<img src="./Final Project Documentation/Final Project Documentation-1.svg">
<img src="./Final Project Documentation/Final Project Documentation-2.svg">
<img src="./Final Project Documentation/Final Project Documentation-3.svg">
<img src="./Final Project Documentation/Final Project Documentation-4.svg">
<img src="./Final Project Documentation/Final Project Documentation-5.svg">
<img src="./Final Project Documentation/Final Project Documentation-6.svg">
<img src="./Final Project Documentation/Final Project Documentation-7.svg">
<img src="./Final Project Documentation/Final Project Documentation-8.svg">
<img src="./Final Project Documentation/Final Project Documentation-9.svg">
<img src="./Final Project Documentation/Final Project Documentation-10.svg">
<img src="./Final Project Documentation/Final Project Documentation-11.svg">
<img src="./Final Project Documentation/Final Project Documentation-12.svg">
<img src="./Final Project Documentation/Final Project Documentation-13.svg">
<img src="./Final Project Documentation/Final Project Documentation-14.svg">
<img src="./Final Project Documentation/Final Project Documentation-15.svg">

 
