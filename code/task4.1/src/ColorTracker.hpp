#include <stdio.h>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


struct model {
	Rect box;
    vector<Mat> histograms;
};

struct candidates {
	vector<Rect> boxes;
    vector<double> scores;
};

class ColorTracker{
    private:
        // Variables
        bool _model_initialized;
        model _model;
        vector<bool> _track_type;
        vector<Mat> _color_spaces;

        // functions
        void _init_model(Mat mask);
        void _get_color_space(Mat frame);
        float _get_distance(Rect candidate_box, Mat mask);
        void _generate_candidate(Mat frame);


    public:
        //Constructor
        ColorTracker(Rect gt, int desired_bins, int in_levels, int in_step, vector<bool> type);
        
        // functions
        Rect track(Mat frame);

        //variables
        int candidate_levels;
        int candidate_step;
        int bins;
        int num_candidates;
        candidates frame_candidates;
        
        
};



