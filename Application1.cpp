#include <opencv2\opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <windows.h>
#include <string>

using namespace cv;
using namespace std;

/*
参考视频:
0.58_0.818 area 30 thresh127
原精视频1 area 30 thresh127
死精子7 area 15 thresh55
*/


const string filePath = "0.58_0.818.avi";

string HashValue(Mat& src) {
	Mat image = src.clone();
	string rst(64, '\0');
	resize(image, image, Size(8, 8));
	int average = mean(image).val[0];
	Mat mask = (image >= (uchar)average);
	uchar* pData;
	int index = 0;
	for (int i = 0; i < mask.rows; i++) {
		pData = mask.ptr<uchar>(i);
		for (int j = 0; j < mask.cols; j++) {
			if (pData[j] == 0) {
				rst[index++] = '0';
			}
			else {
				rst[index++] = '1';
			}
		}
	}
	return rst;
}

int Distance(string& str1, string& str2) {
	if (str1.size() != 64 || str2.size() != 64) {
		return -1;
	}
	int difference = 0;
	for (int i = 0; i < 64; i++) {
		if (str1[i] != str2[i]) {
			difference++;
		}
	}
	return difference;
}

//true为活，false为死
bool judge(string path) {
	VideoCapture video;
	video.open(path);
	Mat frame, cloneFrame;
	string preHash, nowHash;
	video >> frame;

	int frameNumber = 0;
	double distanceCount = 0;
	preHash = HashValue(frame);
	
	while (frameNumber < 20) {
		
		if (frameNumber >= 1) {
			nowHash = HashValue(frame);
			distanceCount += Distance(nowHash, preHash);
			preHash = nowHash;
		}
		video >> frame;
		frameNumber++;
	}
	double lastDifference = distanceCount / frameNumber;
	cout << "图片结果 " << lastDifference << endl;
	if (lastDifference > 1.5) {
		return true;
	}else {
		return false;
	}

}


int main(int argc, char** argv)
{
	VideoCapture capture;
	capture.open(filePath);
	if (!capture.isOpened()) {
		cout << "cannnot open the video, please check the path" << endl;
		getchar();
		getchar();
		return -1;
	}
	Mat frame, cloneFrame; //原始帧
	Mat grayImg, threshImg; //灰度图 二值图
	Mat beforeFrame, nowFrame; //活性计算的帧
	capture >> frame;

	int frameNumber = 0; //开始
	int frameRange = 5; //范围
	double totalEnergy = 0; //精子的总活性

	int thresh = 127; //活精子
	int energyArea = 30; // 判断是完整精子的范围
	//判断视频类别
	bool imgJudge = judge(filePath);
	if (imgJudge == false) {
		thresh = 60;
		energyArea = 15;
	}

	//判断活性相关
	int videoWidth = frame.cols;
	int videoHeight = frame.rows;
	beforeFrame = frame.clone();
	cvtColor(beforeFrame, beforeFrame, COLOR_BGR2GRAY);
	nowFrame = frame.clone();
	cvtColor(nowFrame, nowFrame, COLOR_BGR2GRAY);

	cout << "the size of the image is:" << frame.cols << "*" << frame.rows << endl;
	//视频模式
	//VideoWriter writer;
	//writer.open("deaddemo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT)), 1);

	while (frameNumber < frameRange)
	{
		cloneFrame = frame.clone();
		//转化灰度图
		cvtColor(cloneFrame, grayImg, COLOR_BGR2GRAY);
		string gray_name = filePath + "Y0_gray" + to_string(frameNumber) + ".jpg";
		imwrite(gray_name, grayImg);


		//区域标记
		Mat labels, stats, centroids;
		int nccomps;

		//提高对比度
		if (imgJudge == true) {
			Ptr<CLAHE> clahe = createCLAHE(1.0, Size(8, 8));
			Mat contrastImg;
			clahe->apply(grayImg, contrastImg);
			string contrast_name = filePath + "Y_contrast" + to_string(frameNumber) + ".jpg";
			imwrite(contrast_name, contrastImg);

			//转化二值图
			threshold(contrastImg, threshImg, thresh, 255, THRESH_BINARY_INV);
			string thresh_name = filePath + "Y1_thresh" + to_string(frameNumber) + ".jpg";
			imwrite(thresh_name, threshImg);

			//进行一次开运算
			Mat openImg;
			Mat openKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
			morphologyEx(threshImg, openImg, MORPH_OPEN, openKernel, Point(-1, -1), 1);
			string open_name = filePath + "Y3_OPEN" + to_string(frameNumber) + ".jpg";
			imwrite(open_name, openImg);

			nccomps =  connectedComponentsWithStats(openImg, labels, stats, centroids);
		}
		else {
			threshold(grayImg, threshImg, thresh, 255, THRESH_BINARY_INV);
			string thresh_name = filePath + "Y1_thresh" + to_string(frameNumber) + ".jpg";
			imwrite(thresh_name, threshImg);
			cout << "false'";
			nccomps = connectedComponentsWithStats(threshImg, labels, stats, centroids);
		}


		Mat resultImg = frame.clone();
		int nums = 0;
		//dValue为两帧差值
		Mat dValue;
		dValue = Mat::zeros(Size(videoWidth, videoHeight), CV_8UC1);
		//活性计算
		if (frameNumber >= 1) {
			nowFrame = frame.clone();
			cvtColor(nowFrame, nowFrame, COLOR_BGR2GRAY);

			for (int row = 0; row < videoHeight; row++) {
				for (int col = 0; col < videoWidth; col++) {
					dValue.at<uchar>(row, col) = beforeFrame.at<uchar>(row, col) - nowFrame.at<uchar>(row, col);
					if ((dValue.at<uchar>(row, col) >= 0 && dValue.at<uchar>(row, col) < 20) || (dValue.at<uchar>(row, col) > 235 && dValue.at<uchar>(row, col) <= 255)) {
						dValue.at<uchar>(row, col) = 0;
					}
				}
			}
		}

		double lifeObj = 0; //有活性的目标数
		//对每个connected的结果进行遍历
		for (int i = 1; i < centroids.rows; i++)
		{
			int x = centroids.at<double>(i, 0);
			int y = centroids.at<double>(i, 1);
			Point p(x, y);
			int area = stats.at<int>(i, CC_STAT_AREA);
			
			if (area >= energyArea) {
				//活性相关计算
				if (frameNumber >= 1) {
					double left = stats.at<int>(i, CC_STAT_LEFT); //连通域的boundingbox的最左边
					double top = stats.at<int>(i, CC_STAT_TOP);//连通域的boundingbox的最上边
					double width = stats.at<int>(i, CC_STAT_WIDTH);//连通域的宽
					double height = stats.at<int>(i, CC_STAT_HEIGHT);//连通域的高

					double allPixel = width * height;
					double changePixel = 0;
					for (int row = top; row < top + height; row++) {
						for (int col = left; col < left + width; col++) {
							if (dValue.at<uchar>(row, col) != 0) {
								changePixel++;
							}
						}
					}

					if (changePixel / allPixel >= 0.2) {
						lifeObj++;
					}
					
				}
				//画图
				circle(resultImg, p, 1, Scalar(0, 0, 255), 2, 8);
				nums++;
			}

		}

		cout << "第" << frameNumber << "帧共有" << nums << "个目标" << endl;
		//cout << "活动目标" << lifeObj <<","<< nums <<"结果：" << lifeObj / nums << endl;

		beforeFrame = nowFrame;
		totalEnergy += lifeObj / nums;

		string res_name = filePath + "Y3_result" + to_string(frameNumber) + ".jpg";
		imwrite(res_name, resultImg);

		capture >> frame;
		if (frame.empty()) {
			break;
		}
		frameNumber++;
	}
	cout << "精子活性" << totalEnergy / (frameNumber-1) << endl;
	return 0;
}
