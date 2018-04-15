#include "Tree.h"

Tree::Tree(vector<Mat> &SP, vector<int> &LB, int *ID, int NUM, int NUM_1, int w_w, int maxD, int minL, float minInfo){
	window_width = w_w;

	sample = SP;
	label = LB;
	index = ID;
	count = NUM;
	num_1 = NUM_1;
	num_0 = count - num_1;

	maxDepth = maxD;
	NodeNum = (int)pow(2.0,maxDepth)-1;
	minLeafSample = minL;
	minInfoGain = minInfo;
	node = new Node*[NodeNum];
	node[0] = new Node(sample, label, index, count, num_1, window_width);
	for(int i=1; i<NodeNum; i++)
		node[i] = NULL;
}

Tree::~Tree(){
	/*for(int i=0; i<NodeNum; i++){
		if(node[i] != NULL){
			delete node[i];
			node[i] = NULL;
		}
	}*/

	delete []node;
	node = NULL;
	delete []index;
}

void Tree::train(){
	for(int i=0; i<NodeNum; i++){
		//cout << "*****************************************" <<endl;
		//cout << "i= " << i << endl;
		int parentID = (i-1)/2;
		//cout << "parentID = " << parentID << endl;

		//if parent node is null
		if(node[parentID] == NULL && i != 0){
			//cout << 111 << endl;;
			continue;
		}
		//if parent node is leaf
		if(node[parentID]->isLeaf()){
			//cout << 222 << endl;
			continue;
		}
		//if the left child is out of range
		if(i*2+1>=NodeNum){
			node[i]->setLeaf();
			//cout << "i = " << i << " node set leaf" << endl;
			continue;
		}

		//cout << "Length = " << node[i]->getLength() << endl;
		//randomly choose the patch parameter
		node[i]->select_Para();
		//split the node according to the patch
		node[i]->split_Node();
		//cout << "left length: " << node[i]->get_Left().size() << " right length: " << node[i]->get_Right().size() << endl;
		//calculate the information gain
		//node[i]->calculate_infoGain();
		//cout << "infoGain = " << node[i]->get_infoGain() << endl;

		if(node[i]->get_infoGain() > minInfoGain){
			node[i*2+1] = new Node(sample, label, node[i]->get_Left_index(), node[i]->get_Left_num(), node[i]->get_Left_positive(), window_width);
			node[i*2+2] = new Node(sample, label, node[i]->get_Right_index(), node[i]->get_Right_num(), node[i]->get_Right_positive(), window_width);
			node[i]->release_Vector();
		}
		else{
			//cout << "i = " << i << " node set leaf" << endl;
			node[i]->setLeaf();
		}
	}
	//cout << "Train completed "<< endl;
}

int Tree::predict(Mat test_img){
	int i=0;
	while(!node[i]->isLeaf())
		//cout << "current node = " << i << endl;
		i = 2*i+node[i]->predict(test_img);
	
	//cout << "use judge of " << i << " ndoe" << endl;
	return node[i]->judge(num_1, num_0);
}
