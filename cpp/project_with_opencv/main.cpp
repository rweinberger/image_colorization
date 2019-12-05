#include <cassert>
#include <ctime>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Mat colorize(Mat bw, Mat user_colored, float intensity_threshold, int size_window) {

    Mat diff = cv::abs(bw - user_colored);
    Mat diff_summed;
    cout << cv::sum(diff) << endl;
    cv::reduce(diff, diff_summed, 1, REDUCE_SUM, CV_32F); // not sure if this works??

    return bw; //dummy return

}

// make main
// ./main
int main() {
    Mat bw = imread("./Input/test_bw.png");
    bw.convertTo(bw,CV_32F);
    bw = bw * 1.f/255.f;
    Mat user_colored = imread("./Input/test.png");
    user_colored.convertTo(user_colored,CV_32F);
    user_colored = user_colored * 1.f/255.f;

    // user_colored and bw are exactly same as matlab

    Mat out = colorize(bw, user_colored, .01f, 1);
}

