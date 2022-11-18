#include "FusionTracker.hpp" 
#include<math.h>	

/* Constructor
* Defines the bounding box for ground truth as model.box
* Defines candidate parameters
* DEfines color and gradient histograms if number of bins is higher thn0
*/
FusionTracker::FusionTracker(Rect gt, int in_levels, int in_step, int cbins, vector<bool> type, int gbins) {
    
    _model.box = gt;                                                            
    candidate_levels = in_levels;                                                              
    candidate_step = in_step;
    _model_initialized = false;
    
    if(cbins>0){
        color_bins = cbins;
        _track_type = type;
        _colortrack = true;
    }
    else{
        _colortrack = false;
    }

    if(gbins>0){
        _hog_descriptor.nbins = gbins;
        _gradtrack = true;

    }
    else{
        _gradtrack = false;
    }
}



/* Track
* Normalices distances between target and candidates for both color and gradient trackers if activated 
* Searches for the candidate closest to the target and return its bounding box
*/
Rect FusionTracker::track(Mat frame) {
    
    _generate_candidates(frame);
    vector<double> fusion_scores;

    if(_colortrack){normalize(frame_candidates.color_scores, frame_candidates.color_scores, 0, 1, NORM_MINMAX, -1, Mat() );}
    if(_gradtrack == false){normalize(frame_candidates.gradient_scores, frame_candidates.gradient_scores, 0, 1, NORM_MINMAX, -1, Mat() );}
    
    if(_colortrack&&_gradtrack){
        add(frame_candidates.color_scores, frame_candidates.gradient_scores, fusion_scores);
    }
    else{
        if(_colortrack){fusion_scores = frame_candidates.color_scores;}
        if(_gradtrack){fusion_scores = frame_candidates.gradient_scores;}
    }

    int idx = min_element(fusion_scores.begin(),fusion_scores.end()) - fusion_scores.begin();
    _model.box = frame_candidates.boxes[idx];
    return frame_candidates.boxes[idx];
}


/* Candidate Iterator
* If first frame, generates model histogram(s)
* If not, generates candidate positions as x and y values and calls methods that 
* generate histogram(s) and distance between 
* target and candidate histogram(s). It also saves that distance(s) and the bounding box for each candidate
*/
void FusionTracker::_generate_candidates(Mat frame) {

    Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8U);

    frame_candidates.boxes.clear();
    frame_candidates.color_scores.clear();
    frame_candidates.gradient_scores.clear();

    if(_colortrack){
        _color_spaces.clear();
        _get_color_space(frame);
    }

    if(_gradtrack){cvtColor(frame, frame, CV_BGR2GRAY);}
    int numCandidates = 0; 
    
    if(!_model_initialized){
        _model_initialized = true;
        _init_model(frame,mask);
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
                    if(_colortrack){frame_candidates.color_scores.push_back(_get_color_distance(mask, candidate_box));}
                    if(_gradtrack){frame_candidates.gradient_scores.push_back(_get_gradient_distance(frame,candidate_box));}
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
float FusionTracker::_get_color_distance(Mat mask,Rect candidate_box) {
    
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
                calcHist( &_color_spaces[i], nimages, 0, mask, hist_candidate, dimensions, &color_bins, &h_histRange, uniform, accumulate );
            }
            else{
                calcHist( &_color_spaces[i], nimages, 0, mask, hist_candidate, dimensions, &color_bins, &bgrs_histRange, uniform, accumulate );
            }
            normalize(hist_candidate, hist_candidate, 1, 100, NORM_MINMAX, -1, Mat() );      
            score = compareHist(hist_candidate, _model.histograms[i],3);
            scores.push_back(score);
        }            
    }   
    return norm(scores, NORM_L2); 
}



/* HOG tracking
* Obtains the HOG of one candidate according to grayscale 
* Computes the L2 distance between target and candidate histogram
*/
float FusionTracker::_get_gradient_distance(Mat frame,Rect candidate_box){
    
    Mat croped_frame;
    vector<float> temp_descriptors;
    frame(candidate_box).copyTo(croped_frame);
    resize(croped_frame,croped_frame,Size(64,128));
    _hog_descriptor.compute(croped_frame, temp_descriptors);
    return norm(temp_descriptors, _model.descriptors,NORM_L2);
}



/* Initialize model
* Obtains the histogram(s) of region defined by ground truth  
* Returns 0 "distances" for code consistency
*/
void FusionTracker::_init_model(Mat frame, Mat mask) {

//////////////////////////////////////////////////// COLOR HISTOGRAMS
    if(_colortrack){
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
                    calcHist( &_color_spaces[i], nimages, 0, mask, hist, dimensions, &color_bins, &h_histRange, uniform, accumulate );
                }
                else{
                    calcHist( &_color_spaces[i], nimages, 0, mask, hist, dimensions, &color_bins, &bgrs_histRange, uniform, accumulate );
                }
                normalize(hist, hist, 1, 100, NORM_MINMAX, -1, Mat() );
                _model.histograms.push_back(hist);
            }            
            else{
                _model.histograms.push_back(Mat());
            }
        }
    }

/////////////////////////////////////////////////////////// HOG
    if(_gradtrack){
        Mat croped_frame;
        frame(_model.box).copyTo(croped_frame);
        resize(croped_frame,croped_frame,Size(64,128));

        _hog_descriptor.compute(croped_frame, _model.descriptors);
    }
////////////////////////////////////////////////////////////////
    frame_candidates.boxes.push_back(_model.box);
    frame_candidates.gradient_scores.push_back(0);
    frame_candidates.color_scores.push_back(0);
}


// Converts the input frame into multiple color channels according to tracking type for color histogram tracking
void FusionTracker::_get_color_space(Mat frame){

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




