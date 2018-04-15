#include "RandomForest.h"


RandomForest::RandomForest(vector<Mat> img, vector<int> label, int w_w, int t_n, int s_n, int maxD, int minL, float minInfo){
	window_width = w_w;

	imgData.assign(img.begin(), img.end());
	LabelData.assign(label.begin(), label.end());
	//cout << "sum: = " << accumulate(LabelData.begin(),LabelData.end(), 0) << endl;

	tree_num = t_n;
	maxDepth = maxD;
	minLeafSample = minL;
	minInfoGain = minInfo;
	tree = new Tree*[tree_num];

	if(s_n > imgData.size()){
		cout << "Sample size out of range, " << imgData.size() << " sample will be used" << endl;
		sample_num = imgData.size();
	}
	else
		sample_num = s_n;

	for(int i=0; i<imgData.size(); i++){
		Data *d = new Data(imgData[i], LabelData[i]);
		data.push_back(d);
	}
}

RandomForest::~RandomForest(){
	for(int i=0; i<tree_num; i++){
		if(tree[i] != NULL){
			delete tree[i];
			tree[i] = NULL;
		}
	}

	delete[] tree;
	tree = NULL;

	imgData.clear();
	LabelData.clear();
	data.clear();
}

void RandomForest::train(){
	srand(unsigned(time(NULL)));

	for(int i=0; i<tree_num; i++){
		cout << "Start to train the " << i << "th tree" << endl;
		random_shuffle(data.begin(), data.end());

		vector<int> test;
		for(int i=0; i<imgData.size(); i++)
			test.push_back(data[i]->get_Lab());
		
		//cout << "new sum: = " << accumulate(test.begin(),test.end(), 0) << endl;

		vector<Mat> img;
		vector<int> lab;
		for(int j=0; j<sample_num; j++){
			img.push_back(data[j]->get_Img());
			lab.push_back(data[j]->get_Lab());
		}

		//cout << img.size() << endl;
		///cout << lab.size() << endl;
		//cout << accumulate(lab.begin(), lab.end(),0) << endl;
		//cin.get();

		tree[i] = new Tree(img, lab, window_width, maxDepth, minLeafSample, minInfoGain);
		tree[i]->train();
	}
}

float RandomForest::predict(Mat test_img){
	int vote = 0;
	for(int j=0; j<tree_num; j++)
		vote += tree[j]->predict(test_img);
	
	return 1.0*vote/tree_num;
}