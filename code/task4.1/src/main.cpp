/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	This source code belongs to the template (sample program)
 *	provided for the assignment LAB 4 "Histogram-based tracking"
 *
 *	This code has been tested using:
 *	- Operative System: Ubuntu 18.04
 *	- OpenCV version: 3.4.4
 *	- Eclipse version: 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: April 2020
 * Maria Fernanda Herrera, David Savary 
 */
//includes
#include <stdio.h> 								//Standard I/O library
#include <numeric>								//For std::accumulate function
#include <string> 								//For std::to_string function
#include <opencv2/opencv.hpp>					//opencv libraries
#include "utils.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance
#include "ColorTracker.hpp" 							//for functions readGroundTruthFile & estimateTrackingPerformance

//namespaces
using namespace cv;
using namespace std;

//main function
int main(int argc, char ** argv)
{
	if (argc < 2){
		cout << "Missing argument." << endl;
        cout << "Example: ./Lab3.0AVSA2020 path/to/video1.mp4 path/to/video2.mp4" << endl;
        return -1;
	}
	
	////////////////////////////////////////////
	//         TRACKING PARAMETERS
	///////////////////////////////////////////
	int bins = 64;
	int candidate_levels = 5;
	int candidate_step = 1;
	vector<bool> track_type;
	track_type.push_back(false);	// blue
	track_type.push_back(false);	// green
	track_type.push_back(false);	// red
	track_type.push_back(true); // h
	track_type.push_back(false); // s
	track_type.push_back(false); // gray
	////////////////////////////////////////////

	int NumSeq = argc-1;
    cout << "Numvideos: " << NumSeq << endl;

	//PLEASE CHANGE 'dataset_path' & 'output_path' ACCORDING TO YOUR PROJECT
	//std::string dataset_path = "/home/avsa/avsa/lab4/AVSA_lab4_datasets/datasets";									//dataset location.
	std::string output_path = "./outvideos/";									//location to save output videos
    string makedir_cmd = "mkdir " + output_path;
    system(makedir_cmd.c_str());

	// dataset paths
	//std::string sequences[] = {"bolt1",										//test data for lab4.1, 4.3 & 4.5
	//						   "sphere","car1",								//test data for lab4.2
	//						   "ball2","basketball",						//test data for lab4.4
	//						   "bag","ball","road",};						//test data for lab4.6
	
	std::string image_path = "%08d.jpg"; 									//format of frames. DO NOT CHANGE
	std::string groundtruth_file = "groundtruth.txt"; 						//file for ground truth data. DO NOT CHANGE
	//int NumSeq = sizeof(sequences)/sizeof(sequences[0]);					//number of sequences

	//Loop for all sequence of each category
	for (int s=0; s<NumSeq; s++ )
	{
		
		/////////////////////////////
		std::string sequence = argv[s+1];		
		std::string str = to_string(s);
		makedir_cmd = "mkdir "+ output_path + "/Seq_" + str;
		system(makedir_cmd.c_str());
		int it = 0;
		////////////////////////////

		Mat frame;										//current Frame
		int frame_idx=0;								//index of current Frame
		std::vector<Rect> list_bbox_est, list_bbox_gt;	//estimated & groundtruth bounding boxes
		std::vector<double> procTimes;					//vector to accumulate processing times
		
		//std::string inputvideo = dataset_path + "/" + sequences[s] + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		std::string inputvideo = sequence + "/img/" + image_path; //path of videofile. DO NOT CHANGE
		VideoCapture cap(inputvideo);	// reader to grab frames from videofile

		//check if videofile exists
		if (!cap.isOpened())
			throw std::runtime_error("Could not open video file " + inputvideo); //error if not possible to read videofile

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		cv::Size frame_size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT));//cv::Size frame_size(700,460);
		VideoWriter outputvideo(output_path+"outvid_" + str+".avi",CV_FOURCC('X','V','I','D'),10, frame_size);	//xvid compression (cannot be changed in OpenCV)

		//Read ground truth file and store bounding boxes
		std::string inputGroundtruth = sequence + "/" + groundtruth_file;//path of groundtruth file. DO NOT CHANGE
		list_bbox_gt = readGroundTruthFile(inputGroundtruth); //read groundtruth bounding boxes

		//main loop for the sequence
		std::cout << "Displaying sequence at " << inputvideo << std::endl;
		std::cout << "  with groundtruth at " << inputGroundtruth << std::endl;
		
		/////////////////////////////////////////////////////////////////
		ColorTracker ctracker(list_bbox_gt[0],bins,candidate_levels,candidate_step,track_type);

		for (;;) {
			//get frame & check if we achieved the end of the videofile (e.g. frame.data is empty)
			cap >> frame;
			if (!frame.data)
				break;

			//Time measurement
			double t = (double)getTickCount();
			frame_idx=cap.get(cv::CAP_PROP_POS_FRAMES);			//get the current frame

			////////////////////////////////////////////////////////////////////////////////////////////
			//DO TRACKING
			//Change the following line with your own code

		    //cout<<"HIIIIIIIIIIIIIIIIIIIIIIII"<<endl;

			list_bbox_est.push_back(ctracker.track(frame));//we use a fixed value only for this demo program. Remove this line when you use your code
			//...
			// ADD YOUR CODE HERE
			//...
			////////////////////////////////////////////////////////////////////////////////////////////

			//Time measurement
			procTimes.push_back(((double)getTickCount() - t)*1000. / cv::getTickFrequency());
			//std::cout << " processing time=" << procTimes[procTimes.size()-1] << " ms" << std::endl;

			// plot frame number & groundtruth bounding box for each frame
			putText(frame, std::to_string(frame_idx), cv::Point(10,15),FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255)); //text in red
			rectangle(frame, list_bbox_gt[frame_idx-1], Scalar(0, 255, 0));		//draw bounding box for groundtruth
			rectangle(frame, list_bbox_est[frame_idx-1], Scalar(0, 0, 255));	//draw bounding box (estimation)

			//show & save data
			imshow("Tracking for "+sequence+" (Green=GT, Red=Estimation)", frame);
			outputvideo.write(frame);//save frame to output video
			//string outFile1 = output_path + "/Seq_" + str + "/out"+ to_string(it) +".png";
			// bool write_result1 = false;
	        // write_result1 = imwrite(outFile1, frame);
	        // if (!write_result1) printf("ERROR: Can't save fRAME mask.\n");


			//exit if ESC key is pressed
			if(waitKey(30) == 27) break;
			it++;
		}

		//comparison groundtruth & estimation
		vector<float> trackPerf = estimateTrackingPerformance(list_bbox_gt, list_bbox_est);

		//print stats about processing time and tracking performance
		std::cout << "  Average processing time = " << std::accumulate( procTimes.begin(), procTimes.end(), 0.0) / procTimes.size() << " ms/frame" << std::endl;
		std::cout << "  Average tracking performance = " << std::accumulate( trackPerf.begin(), trackPerf.end(), 0.0) / trackPerf.size() << std::endl;

		//release all resources
		cap.release();			// close inputvideo
		outputvideo.release(); 	// close outputvideo
		destroyAllWindows(); 	// close all the windows
	}
	printf("Finished program.");
	return 0;
}
