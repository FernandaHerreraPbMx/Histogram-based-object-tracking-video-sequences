#include <stdio.h>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct model {
	Rect box;
    vector<float> descriptors;
};

struct candidates {
	vector<Rect> boxes;
    vector<double> scores;
};


class GradientTracker{
    private:
        // variables
        bool _rgb;
        bool _model_initialized;
        model _model;
        HOGDescriptor _hog_descriptor;

        // functions
        void _init_model(Mat frame);
        float _get_distance(Mat frame, Rect box);
        void _generate_candiates(Mat frame);

    public:
        // Constructor
        GradientTracker(Rect gt, int bins, int candidate_levels,int candidate_gap);

        // functions
        Rect track(Mat frame);
        
        // variables
        int candidate_levels;
        int candidate_step;
        candidates frame_candidates;
};



