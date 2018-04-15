#include "ExtractData.h"
#include "ReadData.h"
#include "Node.h"
#include "Tree.h"
#include "Data.h"
#include "RandomForest.h"
#include "Evaluate.h"

#include <time.h>

int main(){
	string train_fold = "C:/45 Thesis/data/train/";
	string test_fold = "C:/45 Thesis/data/test/";
	string out_fold = "C:/45 Thesis/data/train/extracted/";

	float train_thresh = 0.35;
	float test_thresh = 0.35;

	int sample_interval = 7;

	bool get_train_set = false;

	cout << "*****************Start to extract sub-image*****************" << endl;
	extractData(train_fold, out_fold, train_thresh, get_train_set);
	cout << "*****************Extraction completed*****************" << endl << endl;

	vector<Mat> imgTrain;
	vector<int> labelTrain;

	//readData(imgList, labelList);
	cout << "*****************Start to read training data*****************" << endl;
	//double start_rd,end_rd;
	//start_rd = clock();
	readTrainData(out_fold, imgTrain, labelTrain);
	//getTrainingSet_new(train_fold, imgTrain, labelTrain, train_thresh);
	//end_rd = clock();
	//cout << "read time = " << end_rd - start_rd << endl;
	cout << "*****************Reading completed*****************" << endl << endl;

	/*cin.get();
	imgTrain.clear();
	cout << "clear completed" << endl;
	cin.get();
	vector<Mat>(imgTrain).swap(imgTrain);
	cout << "swap completed" << endl;
	cin.get();*/

	double start,end;

	for(float i=25; i<=25; i+=5){		
		int window_width = 30;

		int tree_num = i;
		int sample_num = 3000;
		int maxDepth = 20;
		int minLeafSample = 1;
		float minInfo = 0;

		cout << "*****************Start to train the model*****************" << endl;
		start=clock();
		RandomForest *RF = new RandomForest(imgTrain, labelTrain, window_width, tree_num, sample_num, maxDepth, minLeafSample, minInfo);
		RF->train();
		end = clock();
		double train_t = (end - start) / CLOCKS_PER_SEC ;
		cout << "*****************Training completed*****************" << endl << endl;

		for(float j=0.6; j<0.8; j+=0.05){
			cout << "*****************Start to evaluate the performance*****************" << endl;
			start=clock();
			float prob_threshold = j;
			get_predict_result(RF, test_fold, 30, 5, prob_threshold);
			end=clock();
			double test_t = (end - start) / CLOCKS_PER_SEC ;
			cout << "*****************Evaluation completed*****************" << endl << endl;

			cout << "*****************Start to calculate F1 score*****************" << endl;
			float F1_score = get_F1_score(test_fold);
			cout << "*****************Calculation completed*****************" << endl << endl;

			ofstream fin("e:\\45 Thesis\\result\\result.csv",ios::app);
			if(!fin){
				cout << "open file error" <<endl; 
				cin.get();
				return 0;
			}

			fin <<",tree num," <<  tree_num << ",sumple num," << sample_num << ",maxDepth," << maxDepth << ",minLeafSample," << minLeafSample << ",minInfo," << minInfo <<",train time(second)," << train_t << ",test time(second)," <<  test_t <<",window width," << window_width << ",prob threshold," << prob_threshold << endl;
			fin.close();
		}
		
		delete RF;
	}

	cout << "*****************Benchmark completed*****************" << endl;
	cin.get();

	imgTrain.clear();
	labelTrain.clear();

	/*vector<Data*> dd;

	for(int i=0; i<10; i++){
		Data *d = new Data(imgList[i], labelList[i]);
		dd.push_back(d);
	}

	for(int i=0; i<10; i++){
		imshow("", dd[i]->get_Img());
		waitKey(0);
	}*/


	//cout << "row = " << imgList[0].rows << " col = " << imgList[0].cols << endl;
	//cout << "size = " << imgList[0]. << endl;
	//Node *nt = new Node(imgList, labelList);
	//cout << nt->calculate_entropy(labelList) << endl;;

	/*Tree *tree = new Tree(imgList, labelList);
	tree->train();

	cout << tree->predict(imgList[27]) << endl;*/

	//test the infoGain
	/*vector<Mat> tmpImg;
	vector<int> tmpLab;
	for(int i=0; i<imgList.size(); i++){
		tmpImg.push_back(imgList[i]);
		tmpLab.push_back(labelList[i]);
	}*/

	/*Node *test = new Node(imgList, labelList);
	test->select_Para();

	test->split_Node();
	//test->calculate_infoGain();
	cout << "infogain: " << test->get_infoGain() << endl;

	vector<Mat> left = test->get_Left();
	vector<Mat> right = test->get_Right();

	cout << left.size() << " " << right.size() << endl;*/


	//test the data read
	/*for(int i=0; i<10; i++){
		namedWindow("123");
		imshow("123", imgList[i]);
		waitKey(0);
	}*/
}