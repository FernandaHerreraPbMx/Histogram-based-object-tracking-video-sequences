#include <stdio.h>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


struct model {
	Rect box;
    vector<Mat> histograms;
    vector<float> descriptors;
};

struct candidates {
	vector<Rect> boxes;
    vector<double> color_scores;
    vector<double> gradient_scores;
};

class FusionTracker{
    private:
        // Variables
        bool _model_initialized;
        bool _colortrack;
        bool _gradtrack;
        model _model;
        HOGDescriptor _hog_descriptor;
        vector<bool> _track_type;
        vector<Mat> _color_spaces;


        // functions
        void _init_model(Mat frame, Mat mask);
        void _get_color_space(Mat frame);
        float _get_color_distance(Mat mask,Rect candidate_box);
        float _get_gradient_distance(Mat frame,Rect candidate_box);
        void _generate_candidates(Mat frame);


    public:
        //Constructor
        FusionTracker(Rect gt, int in_levels, int in_step, int cbins, vector<bool> type, int gbins) ;
        
        // functions
        Rect track(Mat frame);

        //variables
        int candidate_levels;
        int candidate_step;
        int color_bins;
        int num_candidates;
        candidates frame_candidates;
        
        
};



