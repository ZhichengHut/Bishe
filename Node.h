#ifndef NODE_H
#define NODE_H

#include<stdio.h>
#include<vector>
#include<list>
#include <opencv.hpp>

#include <time.h>
#include <math.h>
#include <cmath>
#include<numeric>

using namespace std;
using namespace cv;

class Node{
private:
	int x1;
	int x2;
	int y1;
	int y2;
	int d;
	float theta;
	int sample_num;
	int num_1;
	//int threshold;
	bool LeafNode;
	float infoGain;
	float Entro;
	vector<Mat> imgList;
	vector<int> imgLabel;
	int *index;
	int *left_index;
	int *right_index;
	int count_left;
	int count_right;
	int count_left_1;
	int count_right_1;

public:
	Node(vector<Mat> &sample, vector<int> &label, int *ID, int NUM, int NUM_1, int w_w = 1);
	~Node();

	inline int *get_Left_index(){return left_index;};
	inline int *get_Right_index(){return right_index;};
	inline int get_Left_num(){return count_left;};
	inline int get_Right_num(){return count_right;};
	inline int get_Left_positive(){return count_left_1;};
	inline int get_Right_positive(){return count_right_1;};

	inline void setLeaf(){LeafNode = true;};
	inline bool isLeaf(){return LeafNode;};
	void select_Para();
	//void calculate_infoGain();
	float calculate_entropy(int count, int label_1);
	inline float get_infoGain(){return infoGain;};
	void split_Node();
	void release_Vector();
	int predict(Mat test_img);
	int judge(int num_1 = 550, int num_0 = 2616);
	inline int getLength(){return imgList.size();};
	//void split_new(vector<Mat> leftImg, vector<Mat>rightImg, vector<int>leftLabel, vector<int>rightLable);
};
#endif//NODE_H