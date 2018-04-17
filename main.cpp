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
	cout << "size = " << imgTrain.size() << endl;

	double start,end;

	for(float i=25; i<=25; i+=5){		
		int window_width = 30;

		int tree_num = i;
		int sample_num = 2000;
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

		cout << "*****************Start to evaluate the performance*****************" << endl;
		start=clock();
		get_predict_result(RF, test_fold);
		end=clock();
		double test_t = end - start;
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

		fin <<",tree num," <<  tree_num << ",sumple num," << sample_num << ",maxDepth," << maxDepth << ",minLeafSample," << minLeafSample << ",minInfo," << minInfo <<",train time," << train_t << ",test time," <<  end - start <<",window width," << window_width << endl;
		fin.close();

		/*for(float j=0.6; j<0.8; j+=0.05){
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
		}*/
		
		delete RF;
	}

	cout << "*****************Benchmark completed*****************" << endl;
	cin.get();

	imgTrain.clear();
	labelTrain.clear();
}