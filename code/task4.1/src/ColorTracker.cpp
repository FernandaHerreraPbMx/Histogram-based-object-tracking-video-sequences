#include "ColorTracker.hpp" 
#include<math.h>	

/* Constructor
* Defines the bounding box for ground truth as model.box
* Defines candidate parameters
* DEfines bins  and color channel for color histogram
*/
ColorTracker::ColorTracker(Rect gt, int desired_bins, int in_levels, int in_step, vector<bool> type) {
    
    _model.box = gt;
    bins = desired_bins;

    candidate_levels = in_levels;
    candidate_step = in_step;
    _model_initialized = false;
    _track_type = type;
}

/* Track
* Searches for the candidate closest to the target and return its bounding box
*/
Rect ColorTracker::track(Mat frame) {    
    _generate_candidate(frame);
    int idx = min_element(frame_candidates.scores.begin(),frame_candidates.scores.end()) - frame_candidates.scores.begin();
    _model.box = frame_candidates.boxes[idx];
    return frame_candidates.boxes[idx];
}


/* Candidate Iterator
* If first frame, generates model histogram(s)
* If not, generates candidate positions as x and y values and calls methods that 
* generate histogram(s) and  distance between 
* target and candidate histogram(s). It also saves that distance and the bounding box for each candidate
*/
void ColorTracker::_generate_candidate(Mat frame) {

    frame_candidates.boxes.clear();
    frame_candidates.scores.clear();

    int numCandidates = 0;
    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8U);
    _color_spaces.clear();
    _get_color_space(frame);
    
    if(!_model_initialized){       
        _model_initialized = true;
        _init_model(mask);
    }

    else{
        
        int initialValue = -candidate_levels*candidate_step;
        int finalValue = candidate_levels*candidate_step;

        for (int ix = initialValue; ix <= finalValue; ix = ix + candidate_step){
            for (int iy = initialValue; iy <= finalValue; iy = iy + candidate_step){
                int x = _model.box.x + ix;
                int y = _model.box.y + iy;

                if( (x>=0) && (y>=0) && ( (x+_model.box.width) <= frame.cols ) && ( (y+_model.box.height)<=frame.rows ) ){
                    Rect candidate_box = Rect(x,y,_model.box.width,_model.box.height);                   
                    frame_candidates.boxes.push_back(candidate_box);
                    frame_candidates.scores.push_back(_get_distance(candidate_box,mask));  
                }
            }
        }
    }  
}



/* Color histogram tracking
* Obtains the histogram of one candidate according to the specified channel in track type 
* Computes the Battacharyya distance between target and candidate histogram
* If more than one color channel is specified, the difference distances are mixed using L2 distance
*/
float ColorTracker::_get_distance(Rect candidate_box, Mat mask) {
    
    Mat hist_candidate;
    vector<double> scores;
    double score;
    mask(candidate_box) = 1;

    float bgrsRanges[] = {0,256};
    float hRanges[] = {0,180};
    const float* bgrs_histRange = { bgrsRanges };
    const float* h_histRange = { hRanges };
    bool uniform = true, accumulate = false;
    int nimages = 1,  dimensions = 1;

    for(int i = 0;i < 6; i++){   
        
        if(_track_type[i]){
            if(i==3){
                calcHist( &_color_spaces[i], nimages, 0, mask, hist_candidate, dimensions, &bins, &h_histRange, uniform, accumulate );
            }
            else{
                calcHist( &_color_spaces[i], nimages, 0, mask, hist_candidate, dimensions, &bins, &bgrs_histRange, uniform, accumulate );
            }
            normalize(hist_candidate, hist_candidate, 1, 100, NORM_MINMAX, -1, Mat() );      
            score = compareHist(hist_candidate, _model.histograms[i],3);
            scores.push_back(score);
        }            
    }   
    return norm(scores, NORM_L2); 
}


/* Initialize model
* Obtains the histogram(s) of region defined by ground truth  
* Returns "distances" for code consistency
*/
void ColorTracker::_init_model(Mat mask) {
    
    Mat hist;
    mask(_model.box) = 1;

    float bgrsRanges[] = {0,256};
    float hRanges[] = {0,180};
    const float* bgrs_histRange = { bgrsRanges };
    const float* h_histRange = { hRanges };
    bool uniform = true, accumulate = false;
    int nimages = 1,  dimensions = 1;

    for(int i = 0;i < 6; i++){        
        if(_track_type[i]){

            if(i==3){
                calcHist( &_color_spaces[i], nimages, 0, mask, hist, dimensions, &bins, &h_histRange, uniform, accumulate );
            }
            else{
                calcHist( &_color_spaces[i], nimages, 0, mask, hist, dimensions, &bins, &bgrs_histRange, uniform, accumulate );
            }
            normalize(hist, hist, 1, 100, NORM_MINMAX, -1, Mat() );      
            _model.histograms.push_back(hist);
        }            
        else{
            _model.histograms.push_back(Mat());
        }
    }
    frame_candidates.scores.push_back(0);
    frame_candidates.boxes.push_back(_model.box);
}


// Converts the input frame into multiple color channels according to tracking type for color histogram tracking
void ColorTracker::_get_color_space(Mat frame){

    vector<Mat> bgr_planes;
    split(frame, bgr_planes);
    
    for(int i = 0;i < 3; i++){        
        if(_track_type[i]){
            _color_spaces.push_back(bgr_planes[i]);
            }        
        else{
            _color_spaces.push_back(Mat()); }
    }

    for(int i = 3;i < 5; i++){ 
        if(_track_type[i]){
            Mat hsv;
            vector<Mat> hsv_planes;
            cvtColor(frame, hsv, 40);
            split(hsv, hsv_planes);
            _color_spaces.push_back(hsv_planes[i-3]);
            }      
        else{
            _color_spaces.push_back(Mat());
        }
    }

    if(_track_type[5]){
        Mat gray;
        cvtColor(frame, gray, CV_BGR2GRAY);
        _color_spaces.push_back(gray);
    }
}




