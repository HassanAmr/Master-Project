#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


int outputCounter = 1;
const float ratio_Lowe = 0.8f; // As in Lowe's paper; can be tuned

//Always make sure this is accounted for
const int X_RES = 512;
const int Y_RES = 640;
const int GOOD_PORTION = 10;
const String dataset_location = "/Users/Hassan/Workspace/OpenCV/Dataset/BigBird/";
const String dataset_type = ".jpg";
//const int GOOD_PTS_MAX = 50;
int64 work_begin = 0;
int64 work_end = 0;

static void workBegin()
{
    work_begin = getTickCount();
}

static void workEnd()
{
    work_end = getTickCount() - work_begin;
}

static double getTime()
{
    return work_end /((double)getTickFrequency() )* 1000.;
}

struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

struct SIFTDetector
{
    Ptr<Feature2D> sift;
    SIFTDetector(double hessian = 800.0)
    {
        sift = SIFT::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        sift->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};
/*
template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};
*/
struct FirstColumnOnlyCmp
{
    bool operator()(const std::vector<int>& lhs, 
                    const std::vector<int>& rhs) const
    {
        return lhs[1] > rhs[1];
    }
};


static Mat drawGoodMatches(
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    const Mat& img1,
    const Mat& img2,
    std::vector<DMatch>& good_matches
    )
{
    // drawing the results
    Mat img_matches;

    drawMatches( img1, keypoints1, img2, keypoints2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );

    return img_matches;
}

//later should be void again, and Mat should go to the new function that will be specifically responsible for homography
Mat findGoodMatches(
    int cols, int rows,//the columns and rows that cover exactly how big the object is, used for RANSAC homography
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector< std::vector<DMatch> >& matches,
    std::vector<DMatch>& backward_matches,
    std::vector<DMatch>& selected_matches
    )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches;
    double minDist = matches.front()[0].distance;
    double maxDist = matches.back()[0].distance;

    for( int i = 0; i < matches.size(); i++ )
    {
        if (matches[i][0].distance < ratio_Lowe * matches[i][1].distance)
        {
            DMatch forward = matches[i][0]; 
            DMatch backward = backward_matches[forward.trainIdx]; 
            if(backward.trainIdx == forward.queryIdx)
            {
                good_matches.push_back(forward);
            }
            
        }
        //good_matches.push_back( matches[i][0] );
        //good_matches.push_back( matches[i] );
    }
    std::cout<<good_matches.size() << "    ";
    //std::cout << "\nMax distance: " << maxDist << std::endl;
    //std::cout << "Min distance: " << minDist << std::endl;

    //std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

//-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }


    std::vector<Point2f> obj_corners(4);
    std::vector<Point2f> scene_corners(4);

    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    Mat H;
    if (obj.size() > 0) 
    {    
        //-- Get the corners from the image_1 ( the object to be "detected" )
        
        obj_corners[0] = Point(0,0);
        obj_corners[1] = Point( cols, 0 );
        obj_corners[2] = Point( cols, rows );
        obj_corners[3] = Point( 0, rows );

        H = findHomography( obj, scene, RANSAC, 3 );

        //std::cout << H << ": Not a proper match. " << good_matches.size() << std::endl;

        if (countNonZero(H) < 1) 
        {
            std::cout << outputCounter++<< ": Not a proper match. " << selected_matches.size() << std::endl;
        }
        else
        {
            perspectiveTransform( obj_corners, scene_corners, H);

            //find out later what this does
            //scene_corners_ = scene_corners;

            //Mat drawing = Mat::zeros( img2.size(), img2.type() );
            //using searchImg since img2 is currently not available, and both are the same size.
            //later should be set to the region where the object surely is.
            Mat drawing = Mat::zeros( Y_RES, X_RES, CV_8UC1);

            line( drawing,
                scene_corners[0], scene_corners[1],
                Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[1], scene_corners[2],
                Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[2], scene_corners[3],
                Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[3], scene_corners[0],
                Scalar( 255 ), 3, 8 );

            //find contours of the above drawn region
            findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            double val = 0.0;

            if (contours.size() > 0)
            {
                for( size_t i = 0; i < good_matches.size(); i++ )
                {
                    val = pointPolygonTest( contours[0], keypoints2[ good_matches[i].trainIdx ].pt , false );

                    if (val >= 0)
                    {
                        selected_matches.push_back(good_matches[i]);
                    }
                }
            }

            if (selected_matches.size() > 0)
            {
                std::cout << outputCounter++<< ": A proper match. " << selected_matches.size()  << std::endl;               
            }
            else
            {
                std::cout << outputCounter++<< ": Not a proper match. " << selected_matches.size()  << std::endl;
            } 
        }
    }
    else
    {
        std::cout << outputCounter++<< ": Not a proper match. " << selected_matches.size()  << std::endl;
    }
    return H;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    const char* keys =
        "{ h help     |                  | print help message  }"
        "{ t test     | test.jpg          | specify left image  }"
        "{ m cpu_mode |                  | run without OpenCL }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        std::cout << "Usage: surf_matcher [options]" << std::endl;
        std::cout << "Available options:" << std::endl;
        cmd.printMessage();
        return EXIT_SUCCESS;
    }
    if (cmd.has("cpu_mode"))
    {
        ocl::setUseOpenCL(false);
        std::cout << "OpenCL was disabled" << std::endl;
    }

    UMat img1, queryImg;

    std::string testName = cmd.get<std::string>("t");
    std::cout << "Test = " <<testName<< std::endl;

    imread(testName, CV_LOAD_IMAGE_GRAYSCALE).copyTo(queryImg);
    if(queryImg.empty())
    {
        std::cout << "Couldn't load " << testName << std::endl;
        cmd.printMessage();
        return EXIT_FAILURE;
    }

    //crop scrImg into img1
    // Setup a rectangle to define your region of interest
    int imgHeight = queryImg.size().height;
    int imgWidth = queryImg.size().width;
    //the following values should come from a region detection algorithm
    int start_X = imgWidth/3;
    int start_Y = imgHeight/4;
    int width_X = imgWidth/3;
    int width_Y = imgHeight/2;
    Rect myROI(start_X, start_Y, width_X, width_Y);

    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    img1 = queryImg(myROI);


    //declare input/output
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector< std::vector<DMatch> > matches;
    std::vector<DMatch> backward_matches;

    UMat _descriptors1, _descriptors2;
    Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
        descriptors2 = _descriptors2.getMat(ACCESS_RW);

    //instantiate detectors/matchers
    SURFDetector surf;    
    //SIFTDetector sift;

    //SURFMatcher<BFMatcher> matcher;
    BFMatcher matcher;
    
    
    surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
    //sift(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);

    String dscspath = "/Users/Hassan/Workspace/OpenCV/Dataset/db/Desciptors.xml";
    String kptspath = "/Users/Hassan/Workspace/OpenCV/Dataset/db/Keypoints.xml";
    String idspath = "/Users/Hassan/Workspace/OpenCV/Dataset/db/Mapping_IDs.xml";

    FileStorage dscs(dscspath, cv::FileStorage::READ);
    FileStorage kpts(kptspath, cv::FileStorage::READ);
    FileStorage ids(idspath, cv::FileStorage::READ);

    int matchesFound = 0;

    std::vector< std::vector<DMatch> > final_matches;

    std::vector<Mat> hMatrices;
    std::vector< std::vector<int> > ranked_IDs;
    int cols = img1.cols;
    int rows = img1.rows;
    String curr_img;
    for (int i = 1; i <= 600; i++)
    {
        //load descriptors2
        //surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
        cv::String iValue = std::to_string(i);
        cv::String filename = "node_" + iValue;
        kpts[filename] >> keypoints2;
        dscs[filename] >> descriptors2;
        ids[filename] >> curr_img;
        std::cout << curr_img << " -> ";
        
        //std::vector<DMatch> matches;
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);// Find two nearest matches
        matcher.match(descriptors2, descriptors1, backward_matches);
        
        std::vector<DMatch> selected_matches;

        hMatrices.push_back( findGoodMatches(cols, rows, keypoints1, keypoints2, matches, backward_matches, selected_matches) );
        
        final_matches.push_back(selected_matches);
        matchesFound = selected_matches.size();

        //do a data structure for holding IDs with rank, rank being the number found just above.
        //push to this data structure
        //after this loop, this data structure should be sorted in descending order, and the selected amount should be filtered from it (top 10 for example).
        std::vector<int> currentItem;
        currentItem.push_back(i);
        currentItem.push_back(matchesFound); //rank
        ranked_IDs.push_back(currentItem);
        
        matches.clear();
        backward_matches.clear();
        selected_matches.clear();
        keypoints2.clear();
        descriptors2.release();
    }
    
    dscs.release();
    std::cout << std::endl;

    //-- Sort matches and preserve top 10% matches
    //std::sort(matches.begin(), matches.end());
    std::sort(ranked_IDs.begin(), ranked_IDs.end(), FirstColumnOnlyCmp());
    int currID, currRank;
    std::vector<DMatch> currMatches;
    Mat currH;
    UMat img2;
    String input_file, output_file;
    double currFitnessScore = 0.0;
    for (int i = 0; i < GOOD_PORTION; i++)
    {
        currID = ranked_IDs[i][0];
        currRank = ranked_IDs[i][1];
        currMatches = final_matches[currID - 1];    //minus 1 because IDs start at 1 while index start at 0
        currH = hMatrices[currID - 1];              //minus 1 because IDs start at 1 while index start at 0
        
        cv::String idValue = std::to_string(currID);
        cv::String filename = "node_" + idValue;
        kpts[filename] >> keypoints2;
        ids[filename] >> curr_img;
        input_file = dataset_location + curr_img + dataset_type;
        imread(input_file, CV_LOAD_IMAGE_GRAYSCALE).copyTo(img2);//get corresponding image

        currFitnessScore = (double)currMatches.size()/ (double)keypoints2.size();
        std::cout << curr_img << " -> "<< currFitnessScore << std::endl << "H = " << std::endl << currH << std::endl << std::endl;

        //TODO: compute transform matrix and print 
        
        //write image to disk
        Mat img_matches = drawGoodMatches(keypoints1, keypoints2, img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), currMatches);
        while(img_matches.empty()){};
        currFitnessScore *= 100;
        cv::String fitnessValue = std::to_string(currFitnessScore);
        cv::String fitnessText = "Fitness: " + fitnessValue + '%';
        putText(img_matches, fitnessText, Point(5, img1.rows + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        output_file = "output/" + curr_img + dataset_type;
        imwrite(output_file, img_matches);
    }

    kpts.release();
    ids.release();

    return EXIT_SUCCESS;
}
