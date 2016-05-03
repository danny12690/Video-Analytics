#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<iostream>
#include<stdio.h>
#include<string>
#include<sstream>
using namespace cv;
using namespace std;
const int FRAME_HEIGHT = 480;
const int FRAME_WIDTH = 640;
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
const int MAX_NUM_OBJECTS = 50; //Maximum number of objects that can be detected//
//We use this to distinguish between noisy pixel blobs and actual object pixels//
double sizeScale = 0.0; //Scaling factor for training//
double rw = 4.0;
double k = 1800.0;
double k1 = 0.0;
int d = 15;
double ow = 0.0;
//-------------------------The Main Program--------------------------------------------------------------------//
int main(int argc, char** argv)
{
	//--------------------------------------------------------------------------------------------------------//
	//we create HSV track bars and capture webcam image//
	Mat image;
	int x = 0;
	int y = 0;
	//--------------------------------------------------------------------------------------------------//
	VideoCapture cap;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	cap.open(0);
	namedWindow("Adjust detection Color", CV_WINDOW_AUTOSIZE); //create a window called "Adjust detection Color"	
	//The variables for HSV adjustments
	int LowH = 0;
	int iLastX = -1;
	int iLastY = -1;
	int HighH = 179;
	int LowS = 0;
	int HighS = 255;
	int LowV = 0;
	int HighV = 255;

	//We create seek bars in the HSV adjustment window//
	cvCreateTrackbar("LowH", "Adjust detection Color", &LowH, 179);
	cvCreateTrackbar("HighH", "Adjust detection Color", &HighH, 179);
	cvCreateTrackbar("LowS", "Adjust detection Color", &LowS, 255);
	cvCreateTrackbar("HighS", "Adjust detection Color", &HighS, 255);
	cvCreateTrackbar("LowV", "Adjust detection Color", &LowV, 255);
	cvCreateTrackbar("HighV", "Adjust detection Color", &HighV, 255);

	while (1)
	{
		Mat image;
		Mat imgOriginal;
		Mat imgHSV;
		Mat imgThresholded;
		try
		{
			bool b = cap.read(imgOriginal);
			cap.read(image);
		}
		catch (Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: imshow:\n" << err_msg << std::endl;
		}


		//------------------------------------------------------------------------------------------------//
		//HSV conversion and morphological operations//

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
		inRange(imgHSV, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), imgThresholded);
		//morphological opening (remove small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (fill small holes in the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//We need to always show the threshopld image for reference//
		imshow("Thresholded Image", imgThresholded); //show the thresholded image
		//Once adjusted and the Object appears in the thresholded frame, we need to start tracking that object//
		imshow("Original", imgOriginal);

		//-------------------------------------------------------------------------------------------------//
		// Declares a vector of vectors to store the contours
		vector<vector<Point> > v;
		//------------------------------------------------------------------------------------------------//
		//start the detect mode//
		//creates a red bounding box around the object and also turns it blue//
		//------------------------------------------------------------------------------------------------//
		if (waitKey(1) == 'r')
		{
			while (1)
			{
				cap.read(imgOriginal);
				imgOriginal.copyTo(image);
				cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
				inRange(imgHSV, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), imgThresholded);
				//morphological opening (remove small objects from the foreground)
				erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

				//morphological closing (fill small holes in the foreground)
				dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

				imshow("Thresholded Image", imgThresholded); //show the thresholded image
				findContours(imgThresholded, v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				// Finds the contour with the largest area
				int area = 0;
				int idx;
				for (int i = 0; i < v.size(); i++) {
					if (area < v[i].size())
						idx = i;
				}
				//-----------------------------------------------------------------------------------------------//
				// Calculates the bounding rect of the largest area contour
				Rect rect = boundingRect(v[idx]);
				Point pt1, pt2;
				pt1.x = rect.x;
				pt1.y = rect.y;
				pt2.x = rect.x + rect.width;
				pt2.y = rect.y + rect.height;
				//----------------------------------------------------------------------------------------------//
				// Draws the rect in the original image, change the color of the detected object and show it//
				rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
				for (int i = pt1.x; i <= pt2.x; i++)
				{
					for (int j = pt1.y; j <= pt2.y; j++)
					{
						if(imgThresholded.at<uchar>(j,i)!=0)
						image.at<Vec3b>(j,i)[0]=255;
					}
				}
				if(waitKey(1)=='s')
				d = k / rect.width;
				imshow("Original", imgOriginal);
				putText(image,"Reference Width : "+to_string(rw), Point(30, 30), 1, 1, Scalar(0, 0, 255), 1, 8, false);
				imshow("Copied", image);
				if (waitKey(1) == 'd')
				{
						k1 = d * rect.width;
						ow = (float)(rw * k1) / (float)k;
						putText(imgOriginal, "Object Width : " + to_string(ow), Point(30, 30), 1, 1, Scalar(0, 0, 255), 1, 8, false);
						imshow("Original", imgOriginal);
				}
				if (waitKey(1) == 'x')
				{
					return 0;
				}
			}
			//------------------------------------------------------------------------------------------------------//
			//We need to show the webcam image even before the detect mode is activated//
			if (waitKey(1) == 27)
				break;
			else
				continue;
		}
	}
}