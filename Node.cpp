#include "Node.h"

Node::Node(vector<Mat> &sample, vector<int> &label, int *ID, int NUM, int NUM_1, int w_w){	
	imgList = sample;
	imgLabel = label;
	index = ID;
	sample_num = NUM;
	num_1 = NUM_1;
	LeafNode = false;
	infoGain = 0;
	Entro = calculate_entropy(sample_num, num_1);
	theta = 0;
	d = w_w;
	//cout << "Entro = " << Entro << endl;
	//cout << "In Node, sample_num = " << sample_num << ", num_1 = " << num_1 << ", Entro = " << Entro << "index[sn] = " << index[sample_num-1] << endl;
}

Node::~Node(){
	delete []index;
}


void Node::select_Para(){
	srand (time(NULL));
	//d = rand() % (min(imgList[0].cols, imgList[0].rows)) + 1;
	//d = 1;
	x1 = rand() % (imgList[0].cols-d+1);
	x2 = rand() % (imgList[0].cols-d+1);
	y1 = rand() % (imgList[0].rows-d+1);
	y2 = rand() % (imgList[0].rows-d+1);

	theta = 0.0;
	//cout << "location: " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << d << endl;

	//threshold = rand() % 256;
	//threshold = 140;
}

void Node::split_Node(){
	//cout << "element # = " << imgList.size() << endl;
	//int pppkkk = accumulate(imgLabel.begin() , imgLabel.end() , 0);
	//cout << "#1 = " << pppkkk << endl;
	if(num_1 == 0 || num_1==sample_num)
		return;

	srand (time(NULL));

	int r = imgList[0].rows;
	int c = imgList[0].cols;

	for(int i=0; i<100; i++){
		//int d_tmp = rand() % (min(imgList[0].cols, imgList[0].rows)) + 1;
		int d_tmp = d;
		int x1_tmp = rand() % (c-d_tmp+1);
		int y1_tmp = rand() % (r-d_tmp+1);
		int x2_tmp = rand() % (c-d_tmp+1);
		int y2_tmp = rand() % (r-d_tmp+1);

		//cout << "tmp : " << x1_tmp << " " << y1_tmp << " "  << x2_tmp << " "  << y2_tmp << " "  << " " << d_tmp << endl;

		//vector<Mat> leftImg_tmp, rightImg_tmp;
		//vector<int> leftLabel_tmp, rightLabel_tmp;
		//leftLabel_tmp.resize(imgLabel.size());
		//rightLabel_tmp.resize(imgLabel.size());
		//leftLabel_tmp.reserve(imgLabel.capacity());
		//rightLabel_tmp.reserve(imgLabel.capacity());
		//cout << "tmp size: " << leftLabel_tmp.size() << " " << rightLabel_tmp.size() << endl;
		//cin.get();

		int ss_index1 = rand() % sample_num;
		//cout << "ss1 = " << index[ss_index1] << endl;
		float theta_tmp = mean(imgList[index[ss_index1]](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[index[ss_index1]](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
		while(true){
			int ss_index2 = rand() % sample_num;
			if(imgLabel[index[ss_index1]] + imgLabel[index[ss_index2]] == 1){
				//cout << "ss2 = " << index[ss_index2] << endl;
				theta_tmp += (mean(imgList[index[ss_index2]](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0] - mean(imgList[index[ss_index2]](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0]);
				theta_tmp /= 2.0;
				break;
			}
		}

		int left_count = 0;
		int left_1 = 0;
		int right_count = 0;
		int right_1 = 0;

		for(int p=0; p<sample_num; p++){
			//cout << "img size " << imgList[p].cols <<" " << imgList[p].rows << endl;

			float mean1 = mean(imgList[index[p]](Rect(x1_tmp,y1_tmp,d_tmp,d_tmp)))[0];
			//cout << 11 << endl;
			float mean2 = mean(imgList[index[p]](Rect(x2_tmp,y2_tmp,d_tmp,d_tmp)))[0];
			//cout << 22 << endl;

			if(mean1-mean2>theta_tmp){
				//leftImg_tmp.push_back(imgList[p]);
				//leftLabel_tmp.push_back(imgLabel[p]);
				left_count++;
				left_1 += imgLabel[index[p]];
			}
			else{
				//rightImg_tmp.push_back(imgList[p]);
				//rightLabel_tmp.push_back(imgLabel[p]);
				right_count++;
				right_1 += imgLabel[index[p]];
			}
		}

		//cout << "entro = " << Entro <<", length = " << imgLabel.size() << ", 1= " << accumulate(imgLabel.begin(), imgLabel.end(),0) << endl;
		//cout << "left E = " << calculate_entropy(leftLabel_tmp) << ", left length = " << leftLabel_tmp.size() << ", 1 = " <<accumulate(leftLabel_tmp.begin(), leftLabel_tmp.end(),0) << endl;
		//cout << "right E = " << calculate_entropy(rightLabel_tmp) << ", right length = " << rightLabel_tmp.size() << ", 1 = " <<accumulate(rightLabel_tmp.begin(), rightLabel_tmp.end(),0) << endl;
			

		float infoGain_new = Entro - (left_count*calculate_entropy(left_count, left_1) + right_count*calculate_entropy(right_count, right_1))/sample_num;
		
		//cout << "new gain = " << infoGain_new << endl;
		//cin.get();

		if(infoGain_new > infoGain){	
			//cout << "tmp : " << x1_tmp << " " << y1_tmp << " "  << x2_tmp << " "  << y2_tmp << " "  << " " << d << endl;
			infoGain = infoGain_new;
			x1 = x1_tmp;
			x2 = x2_tmp;
			y1 = y1_tmp;
			y2 = y2_tmp;
			d = d_tmp;
			theta = theta_tmp;
			
			//cout << "new gain = " << infoGain << endl;
			
		}
		//cout << "tmp size: " << leftLabel_tmp.size() << " " << rightLabel_tmp.size() << endl;
	}

	int *index_left_tmp = new int[sample_num];
	int *index_right_tmp = new int[sample_num];
	count_left = 0;
	count_left_1 = 0;
	count_right = 0;
	count_right_1 = 0;

	for(int p=0; p<sample_num; p++){
		//cout << "location: " << x1 << " " << y1 << " " << x2 << " " << y2 << " " << d << endl;
		float mean1 = mean(imgList[index[p]](Rect(x1,y1,d,d)))[0];
		//cout << 33 << endl;
		float mean2 = mean(imgList[index[p]](Rect(x2,y2,d,d)))[0];
		//cout << 44 << endl;
		if(mean1-mean2>theta){
			index_left_tmp[count_left] = index[p];
			count_left++;
			count_left_1 += imgLabel[index[p]];
		}
		else{
			index_right_tmp[count_right] = index[p];
			count_right++;
			count_right_1 += imgLabel[index[p]];
		}
	}

	left_index = new int[count_left];
	memcpy(left_index, index_left_tmp, count_left*sizeof(int));
	delete []index_left_tmp;

	right_index = new int[count_right];
	memcpy(right_index, index_right_tmp, count_right*sizeof(int));
	delete []index_right_tmp;
}

void Node::release_Vector(){
	delete []index;
	imgList.clear();
	vector<Mat>().swap(imgList);
	imgLabel.clear();
	vector<int>().swap(imgLabel);
	//delete []left_index;
	//delete []right_index;
}

int Node::predict(Mat test_img){
	//cout << "node predict" << endl; 
	//cout << "x1=" << x1 << "y1=" << y1 << "x2=" << x2 << "y2=" << y2 << "d=" << d << endl;
	float mean1 = mean(test_img(Rect(x1,y1,d,d)))[0];
	float mean2 = mean(test_img(Rect(x2,y2,d,d)))[0];
	if(mean1-mean2>theta){
		//cout << "left node" << endl;
		return 1;
	}
	else{
		//cout << "right node" << endl;
		return 2;
	}
}

int Node::judge(int num_positive, int num_negative){
	//float p_1 = 1.0 * accumulate(imgLabel.begin(), imgLabel.end(),0) / num_positive;
	//float p_2 = 1.0 * (imgLabel.size()-accumulate(imgLabel.begin(), imgLabel.end(),0)) / num_negative;

	int p_1 = num_1;
	int p_2 = sample_num - p_1;

	if(p_2 > p_1)
		return 0;
	else
		return 1;
}

float Node::calculate_entropy(int count, int label_1){
	float entropy = 0;

	if(count != 0){
		float pp = 1.0 * label_1 / count;						//positive%
		float np = 1.0 * (count - label_1) / count;		//negtive%

		if(pp!=0 && np !=0)
			entropy = -1.0*pp*log(1.0*pp)/log(2.0) - 1.0*np*log(1.0*np)/log(2.0);
	}
	return entropy;
}

/*float Node::calculate_entropy(vector<int> label){
	float entropy = 0;

	if(label.size() != 0){
		int class_1 = accumulate(label.begin(), label.end(),0);
		float pp = 1.0 * class_1 / label.size();						//positive%
		float np = 1.0 * (label.size() - class_1) / label.size();		//negtive%

		if(pp!=0 && np !=0)
			entropy = -1.0*pp*log(1.0*pp)/log(2.0) - 1.0*np*log(1.0*np)/log(2.0);
	}
	return entropy;
}*/