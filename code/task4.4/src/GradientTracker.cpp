#include "GradientTracker.hpp" 
#include<math.h>
#include <unistd.h>
#include <iostream>
#include <unistd.h>



/* Constructor
* Defines the bounding box for ground truth as model.box
* Defines candidate parameters
* DEfines HOG bins
*/
GradientTracker::GradientTracker(Rect gt, int bins, int in_cand_levels,int in_cand_step) {
    
    candidate_levels = in_cand_levels;
    candidate_step = in_cand_step;

    _hog_descriptor.nbins = bins;
    _model.box = gt;
    _model_initialized = false;

}


/* Track
* Searches for the candidate closest to the target and return its bounding box
*/
Rect GradientTracker::track(Mat frame) {
    
    _generate_candiates(frame);
    int idx = min_element(frame_candidates.scores.begin(),frame_candidates.scores.end()) - frame_candidates.scores.begin();
    _model.box = frame_candidates.boxes[idx];
    return frame_candidates.boxes[idx];
}


/* Candidate Iterator
* If first frame, generates model HOG
* If not, generates candidate positions as x and y values and calls methods that 
* generate HOG and  distance between 
* target and candidate HOG. It also saves that distance and the bounding box for each candidate
*/
void GradientTracker::_generate_candiates(Mat frame){

    frame_candidates.boxes.clear();
    frame_candidates.scores.clear();
    
    cvtColor(frame, frame, CV_BGR2GRAY);

    if(!_model_initialized) {
        _init_model(frame);
        _model_initialized = true;
    }

    else {

        int initialValue = -candidate_levels*candidate_step;
        int finalValue = candidate_levels*candidate_step;

        for (int ix = initialValue; ix <= finalValue; ix = ix + candidate_step){
            for (int iy = initialValue; iy <= finalValue; iy = iy + candidate_step){

                int x = _model.box.x + ix;
                int y = _model.box.y + iy;

                if( (x>=0) && (y>=0) && ( (x+_model.box.width) <= frame.cols ) && ( (y+_model.box.height)<=frame.rows ) ){

                    Rect candidate_box = Rect(_model.box);
                    candidate_box.x = x;
                    candidate_box.y = y;
                    frame_candidates.boxes.push_back(candidate_box);
                    frame_candidates.scores.push_back(_get_distance(frame,candidate_box));
    
                }
            }
        }
    }
}


/* Initialize model
* Obtains the HOG of region defined by ground truth  
* Returns "distances" for code consistency
*/
void GradientTracker::_init_model(Mat frame){
 
    Mat croped_frame;
    frame(_model.box).copyTo(croped_frame);
    resize(croped_frame,croped_frame,Size(64,128));

    _hog_descriptor.compute(croped_frame, _model.descriptors);
    
    frame_candidates.boxes.push_back(_model.box);
    frame_candidates.scores.push_back(0);

}


/* HOG tracking
* Obtains the HOG of one candidate according to grayscale 
* Computes the L2 distance between target and candidate histogram
*/
float GradientTracker::_get_distance(Mat frame,Rect candidate_box){
    
    Mat croped_frame;
    vector<float> temp_descriptors;
    frame(candidate_box).copyTo(croped_frame);
    resize(croped_frame,croped_frame,Size(64,128));

    _hog_descriptor.compute(croped_frame, temp_descriptors);
    return norm(temp_descriptors, _model.descriptors,NORM_L2);

}