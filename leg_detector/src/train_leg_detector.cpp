/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "laser_processor.h"
#include "calc_leg_features.h"

#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/ml.h"

#include "people_msgs/PositionMeasurement.h"
#include "sensor_msgs/LaserScan.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;
using namespace laser_processor;
using namespace ros;

enum LoadType {
    LOADING_NONE, LOADING_POS, LOADING_NEG, LOADING_TEST
};

inline cv::TermCriteria TC(int iters, double eps)
{
    return cv::TermCriteria(cv::TermCriteria::MAX_ITER + (eps > 0 ? cv::TermCriteria::EPS : 0), iters, eps);
}



class TrainLegDetector {
public:
    ScanMask mask_;
    int mask_count_;

    vector< vector<float> > pos_data_;
    vector< vector<float> > neg_data_;
    vector< vector<float> > test_data_;

    cv::Ptr<cv::ml::RTrees> forest;
    float connected_thresh_;

    int feat_count_;

    TrainLegDetector() : mask_count_(0), connected_thresh_(0.06), feat_count_(0) {
    }

    void loadData(LoadType load, char* file) {
        if (load != LOADING_NONE) {
            switch (load) {
                case LOADING_POS:
                    printf("Loading positive training data from file: %s\n", file);
                    break;
                case LOADING_NEG:
                    printf("Loading negative training data from file: %s\n", file);
                    break;
                case LOADING_TEST:
                    printf("Loading test data from file: %s\n", file);
                    break;
                default:
                    break;
            }

            rosbag::Bag bag;
            //ros::record::Player p;
            //if (p.open(file, ros::Time()))
            bag.open(file, rosbag::bagmode::Read);

            std::vector<std::string> topics;
            topics.push_back(std::string("/scan_merged"));

            rosbag::View view(bag, rosbag::TopicQuery(topics));
            mask_.clear();
            mask_count_ = 0;
            switch (load) {
                case LOADING_POS:
                    foreach(rosbag::MessageInstance const m, view)
                    {
                        sensor_msgs::LaserScan::ConstPtr s = m.instantiate<sensor_msgs::LaserScan>();

                        if (mask_count_++ < 20) {
                            mask_.addScan(s);
                        } else {
                            ScanProcessor processor(s, mask_);
                            processor.splitConnected(connected_thresh_);
                            processor.removeLessThan(5);

                            for (list<SampleSet*>::iterator i = processor.getClusters().begin();
                                    i != processor.getClusters().end();
                                    i++)
                                pos_data_.push_back(calcLegFeatures(*i, s));
                        }
                    }
                    break;
                case LOADING_NEG:
                    mask_count_ = 1000; // effectively disable masking
                    foreach(rosbag::MessageInstance const m, view)
                    {
                        sensor_msgs::LaserScan::ConstPtr s = m.instantiate<sensor_msgs::LaserScan>();

                        if (mask_count_++ < 20) {
                            mask_.addScan(s);
                        } else {
                            ScanProcessor processor(s, mask_);
                            processor.splitConnected(connected_thresh_);
                            processor.removeLessThan(5);

                            for (list<SampleSet*>::iterator i = processor.getClusters().begin();
                                    i != processor.getClusters().end();
                                    i++)
                                neg_data_.push_back(calcLegFeatures(*i, s));
                        }
                    }
                    break;
                case LOADING_TEST:
                    foreach(rosbag::MessageInstance const m, view)
                    {
                        sensor_msgs::LaserScan::ConstPtr s = m.instantiate<sensor_msgs::LaserScan>();

                        if (mask_count_++ < 20) {
                            mask_.addScan(s);
                        } else {
                            ScanProcessor processor(s, mask_);
                            processor.splitConnected(connected_thresh_);
                            processor.removeLessThan(5);

                            for (list<SampleSet*>::iterator i = processor.getClusters().begin();
                                    i != processor.getClusters().end();
                                    i++)
                                test_data_.push_back(calcLegFeatures(*i, s));
                        }
                    }
                    break;
                default:
                    break;
            }
            bag.close();

        }
    }

//    void loadCb(string name, sensor_msgs::LaserScan* scan, ros::Time t, ros::Time t_no_use, void* n) {
//        vector< vector<float> >* data = (vector< vector<float> >*)(n);
//        if (mask_count_++ < 20) {
//            mask_.addScan(scan);
//        } else {
//            ScanProcessor processor(*scan, mask_);
//            processor.splitConnected(connected_thresh_);
//            processor.removeLessThan(5);
//
//            for (list<SampleSet*>::iterator i = processor.getClusters().begin();
//                    i != processor.getClusters().end();
//                    i++)
//                data->push_back(calcLegFeatures(*i, *scan));
//        }
//    }

    void train() {

        int sample_size = pos_data_.size() + neg_data_.size();
        feat_count_ = pos_data_[0].size();

        cv::Mat cv_data = cv::Mat(sample_size, feat_count_, CV_32F);
        cv::Mat cv_resp = cv::Mat(sample_size, 1, CV_32S);


        for (size_t i = 0; i < pos_data_.size(); i++) {
            for (size_t j = 0; j < pos_data_[i].size(); j++) {
                cv_data.at<float>(i,j) = pos_data_[i][j];
            }
            cv_resp.at<int>(i) = 1;
        }

        for (size_t i = 0; i < neg_data_.size(); i++) {
            for (size_t j = 0; j < neg_data_[i].size(); j++) {
                cv_data.at<float>(pos_data_.size() + i ,j) = neg_data_[i][j];
            }
            cv_resp.at<int>(pos_data_.size() + i) = -1;
        }

        cv::Ptr<cv::ml::TrainData> tdata;

        cv::Mat sample_idx = cv::Mat::zeros( 1, cv_data.rows, CV_8U );
        cv::Mat train_samples = sample_idx.colRange(0, sample_size);
        train_samples.setTo(cv::Scalar::all(1));

        int nvars = cv_data.cols;
        cv::Mat var_type( nvars + 1, 1, CV_8U );
        var_type.setTo(cv::Scalar::all(cv::ml::VAR_ORDERED));
        var_type.at<uchar>(nvars) = cv::ml::VAR_CATEGORICAL;

        tdata = cv::ml::TrainData::create(cv_data, cv::ml::ROW_SAMPLE, cv_resp, cv::noArray(), sample_idx, cv::noArray(), var_type);

        forest = cv::ml::RTrees::create();
        forest->setMaxDepth(10);
        forest->setMinSampleCount(10);
        forest->setRegressionAccuracy(0);
        forest->setUseSurrogates(false);
        forest->setMaxCategories(15);
        forest->setPriors(cv::Mat());
        forest->setCalculateVarImportance(true);
        forest->setActiveVarCount(4);
        forest->setTermCriteria(TC(100,0.01f));
        forest->train(tdata);

    }

    void test() {
        cv::Mat tmp_mat = cv::Mat(1, feat_count_, CV_32F);

        int pos_right = 0;
        int pos_total = 0;
        for (size_t i = 0; i < pos_data_.size(); i++) {
            for (size_t j = 0; j < pos_data_[i].size(); j++) {
                tmp_mat.at<float>(j) = pos_data_[i][j];
            }
            if (forest->predict(tmp_mat) > 0)
                pos_right++;
            pos_total++;
        }

        int neg_right = 0;
        int neg_total = 0;
        for (size_t i = 0; i < neg_data_.size(); i++) {
            for (size_t j = 0; j < neg_data_[i].size(); j++) {
                tmp_mat.at<float>(j) = neg_data_[i][j];
            }
            if (forest->predict(tmp_mat) < 0)
                neg_right++;
            neg_total++;
        }


        int test_right = 0;
        int test_total = 0;
        cout << test_data_.size() << endl;
        for (size_t i = 0; i < test_data_.size(); i++) {
            for (size_t j = 0; j < test_data_[i].size(); j++) {
                tmp_mat.at<float>(j) = test_data_[i][j];
            }
            if (forest->predict(tmp_mat) > 0)
                test_right++;
            test_total++;
        }

        printf(" Pos train set: %d/%d %g\n", pos_right, pos_total, (float) (pos_right) / pos_total);
        printf(" Neg train set: %d/%d %g\n", neg_right, neg_total, (float) (neg_right) / neg_total);
        printf(" Test set:      %d/%d %g\n", test_right, test_total, (float) (test_right) / test_total);

    }

    void save(char* file) {
        forest->save(file);
    }
};

int main(int argc, char **argv) {
    TrainLegDetector tld;

    LoadType loading = LOADING_NONE;

    char save_file[100];
    save_file[0] = 0;

    printf("Loading data...\n");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--train"))
            loading = LOADING_POS;
        else if (!strcmp(argv[i], "--neg"))
            loading = LOADING_NEG;
        else if (!strcmp(argv[i], "--test"))
            loading = LOADING_TEST;
        else if (!strcmp(argv[i], "--save")) {
            if (++i < argc)
                strncpy(save_file, argv[i], 100);
            continue;
        } else {
            printf("load");
            tld.loadData(loading, argv[i]);}
    }

    printf("Training classifier...\n");
    tld.train();

    printf("Evlauating classifier...\n");
    tld.test();

    if (strlen(save_file) > 0) {
        printf("Saving classifier as: %s\n", save_file);
        tld.save(save_file);
    }
}
