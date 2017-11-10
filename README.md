# vertical-finger-detection
Computer vision project in detecting a shush face (vertical finger in front of mouth)

Files included:

Part 1:
- DetectSilence.cpp
- DetectSilence
- Mouth.xml
- Nariz.xml (nose cascade from http://alereimondo.no-ip.org/OpenCV/34)
- SilenceImages (images provided for use in training)

Part 2:
- cascade.xml
- pos (positive images from CreateSamples)
- pos-pad (padded vertical finger images)
- neg (negative images)
- neg.txt 
- crop.txt
- cropped.vec

Part 1: Silence

Instead of searching for the mouth in the lower half of the face, I used a nose detector, then tried to look for the mouth in the area below the nose. The nose detector worked sometimes, but overall did not prove to be very accurate. I set up the project so that if a nose was detected, the lower face would be defined as the area below the nose, otherwise, the bottom half of the face would be used. 

This helped a little, but I also noticed that often the mouth detector would consider each corner of the mouth to be a separate mouth, such as in the case of a vertical finger in front of the mouth. I set up a parameter such that a detected "mouth" had to be at least a quarter as wide as the width of the face area in which it was being detected. This helped a lot to get rid of false positive mouth detections. 

Finally, I also experimented with different combinations of the equilizeHist, medianBlur, and blur functions, and different parameters within those functions, and ended up enabling equalizeHist. This helped with some and made some worse, but most importantly it caused face detection to be more accurate in a handful of examples. 

Part 2: Vertical Finger Detection

A total of 30 vertical finger examples were cropped and padded (provided in pos-pad), but only 01-pad.bmp-10-pad.bmp were used for training purposes, due to instructions given in class. 

A total of 10 negative images were used (provided in neg), and each was repeated 10 times for a total of 100 lines in neg.txt. 
There are a total of 1000 positive sample images in pos. 

Parameters of TrainCascade used to generate my detector:

TrainCascade -data . -vec cropped.vec -bg neg.txt -w 10 -h 20 -numPos 800 0 -numStages 18
