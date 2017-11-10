#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;

string OPENCV_ROOT = "/Users/delilalee/Downloads/opencv-3.1.0/";
string cascades = OPENCV_ROOT + "data/haarcascades_cuda/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";

/*  The mouth cascade is assumed to be in the local folder */
string MOUTH_CASCADE_NAME = "Mouth.xml";
// nose case is also assumed to be in the local folder
// Nariz.xml obtained from http://alereimondo.no-ip.org/OpenCV/34 Nose25x15.zip
string NOSE_CASCADE_NAME = "Nariz.xml";

void drawEllipse(Mat frame, const Rect rect, int r, int g, int b) {
    int width2 = rect.width/2;
    int height2 = rect.height/2;
    Point center(rect.x + width2, rect.y + height2);
    ellipse(frame, center, Size(width2, height2), 0, 0, 360,
            Scalar(r, g, b), 2, 8, 0 );
}


bool detectSilence(Mat frame, Rect face, Point location, Mat ROI, CascadeClassifier cascade1)
{
    // frame,location are used only for drawing the detected mouths
    vector<Rect> mouths;
    cascade1.detectMultiScale(ROI, mouths, 1.1, 3, 0, Size(20, 20));
    
    int nmouths = (int)mouths.size();
    int nmouths2 = nmouths;
    for( int i = 0; i < nmouths ; i++ ) {
        Rect mouth_i = mouths[i];
        
        // if a detected mouth is smaller than 1/4 of the face width,
        // it is not counted as a detected mouth
        if(mouth_i.width < (face.width/4)) {
            nmouths2--;
        }
        else {
            drawEllipse(frame, mouth_i + location, 255, 255, 0);
        }
    }
    return(nmouths2 == 0);
}

// you need to rewrite this function
int detect(Mat frame,
           CascadeClassifier cascade_face, CascadeClassifier cascade_mouth,
           CascadeClassifier cascade_nose) {
    Mat frame_gray;
    vector<Rect> faces;
    
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    
    equalizeHist(frame_gray, frame_gray); // input, outuput
    //  medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
    //  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
    /*  input,output,neighborood_size,center_location (neg means - true center) */
    
    cascade_face.detectMultiScale(frame_gray, faces, 1.06, 6, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
    
    /* frame_gray - the input image
     faces - the output detections.
     1.1 - scale factor for increasing/decreasing image or pattern resolution
     3 - minNeighbors.
     larger (4) would be more selective in determining detection
     smaller (2,1) less selective in determining detection
     0 - return all detections.
     0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
     Size(30, 30)) - size in pixels of smallest allowed detection
     */
    
    int detected = 0;
    
    int nfaces = (int)faces.size();
    for( int i = 0; i < nfaces ; i++ ) {
        Rect face = faces[i];
        drawEllipse(frame, face, 255, 0, 255);
        
        // detect noses within the face area
        Rect face_i = Rect(face.x, face.y, face.width, face.height);
        Mat face_i_mat = frame_gray(face_i);
        vector<Rect> noses;
        cascade_nose.detectMultiScale(face_i_mat, noses, 1.1, 3, 0, Size(0,0), Size(30, 30));
        bool nose_found =  false;
        Rect nose;
        int nnoses = (int)noses.size();
        if(nnoses != 0) {
            nose_found = true;
        }
        for(int j=0; j<nnoses; j++) {
            Rect n = noses[j];
            nose = n;
        }
        
        // if a nose is found, run detectSilence on the area of the face below the nose
        // otherwise, run detectSilence on the bottom half of the face
        int x1, y1;
        Rect lower_face;
        if(nose_found) {
            Rect n = Rect((face.x + nose.x), (face.y + nose.y), nose.width, nose.height);
            drawEllipse(frame, n, 0, 255, 255); // yellow
            
            x1 = face.x;
            y1 = face.y + nose.y + (int)((float)nose.height/2);
            
            lower_face = Rect(x1, y1, face.width, (face.height - nose.y - (int)((float)nose.height/2)));
        }
        else {
            x1 = face.x;
            y1 = face.y + face.height/2;
            lower_face =  Rect(x1, y1, face.width, face.height/2);
        }
        
        drawEllipse(frame, lower_face, 100, 0, 255);
        Mat lower_faceROI = frame_gray(lower_face);
        if(detectSilence(frame, face, Point(x1, y1), lower_faceROI, cascade_mouth)) {
            drawEllipse(frame, face, 0, 255, 0);
            detected++;
        }

    }
    return(detected);
}

int runonFolder(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2,
                const CascadeClassifier cascade3,
                string folder) {
    if(folder.at(folder.length()-1) != '/') folder += '/';
    DIR *dir = opendir(folder.c_str());
    if(dir == NULL) {
        cerr << "Can't open folder " << folder << endl;
        exit(1);
    }
    bool finish = false;
    string windowName;
    struct dirent *entry;
    int detections = 0;
    while (!finish && (entry = readdir(dir)) != NULL) {
        char *name = entry->d_name;
        string dname = folder + name;
        Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
        if(!img.empty()) {
            int d = detect(img, cascade1, cascade2, cascade3);
            cerr << d << " detections" << endl;
            detections += d;
            if(!windowName.empty()) destroyWindow(windowName);
            windowName = name;
            namedWindow(windowName.c_str(),CV_WINDOW_AUTOSIZE);
            imshow(windowName.c_str(), img);
            int key = waitKey(0); // Wait for a keystroke
            switch(key) {
                case 27 : // <Esc>
                    finish = true; break;
                default :
                    break;
            }
        } // if image is available
    }
    closedir(dir);
    return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
                const CascadeClassifier cascade2,
                const CascadeClassifier cascade3) {
    VideoCapture videocapture(0);
    if(!videocapture.isOpened()) {
        cerr <<  "Can't open default video camera" << endl ;
        exit(1);
    }
    string windowName = "Live Video";
    namedWindow("video", CV_WINDOW_AUTOSIZE);
    Mat frame;
    bool finish = false;
    while(!finish) {
        if(!videocapture.read(frame)) {
            cout <<  "Can't capture frame" << endl ;
            break;
        }
        detect(frame, cascade1, cascade2, cascade3);
        imshow("video", frame);
        if(waitKey(30) >= 0) finish = true;
    }
}

int main(int argc, char** argv) {
    if(argc != 1 && argc != 2) {
        cerr << argv[0] << ": "
        << "got " << argc-1
        << " arguments. Expecting 0 or 1 : [image-folder]"
        << endl;
        return(-1);
    }
    
    string foldername = (argc == 1) ? "" : argv[1];
    CascadeClassifier faces_cascade, mouth_cascade, nose_cascade;
    
    if(
       !faces_cascade.load(FACES_CASCADE_NAME)
       || !mouth_cascade.load(MOUTH_CASCADE_NAME)
       || !nose_cascade.load(NOSE_CASCADE_NAME)) {
        cerr << FACES_CASCADE_NAME << " or " << MOUTH_CASCADE_NAME
        << " or " << NOSE_CASCADE_NAME
        << " are not in a proper cascade format" << endl;
        return(-1);
    }
    
    int detections = 0;
    if(argc == 2) {
        detections = runonFolder(faces_cascade, mouth_cascade, nose_cascade, foldername);
        cout << "Total of " << detections << " detections" << endl;
    }
    else runonVideo(faces_cascade, mouth_cascade, nose_cascade);
    
    return(0);
}
