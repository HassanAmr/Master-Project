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

const int GOOD_PTS_MAX = 50;
const float GOOD_PORTION = 0.1f;
const float ratio_Lowe = 0.8; // As in Lowe's paper; can be tuned

int64 work_begin = 0;
int64 work_end = 0;

int outputCounter = 1;

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

static Mat findGoodMatches(
    const Mat& img1,
    const Mat& img2,
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector< std::vector<DMatch> >& matches,
    std::vector<DMatch>& backward_matches,
    std::vector<Point2f>& scene_corners_
    )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< DMatch > good_matches, selected_matches;
    double minDist = matches.front()[0].distance;
    double maxDist = matches.back()[0].distance;

    //const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));
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
    

    //std::cout << "\nMax distance: " << maxDist << std::endl;
    //std::cout << "Min distance: " << minDist << std::endl;

    //std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

    // drawing the results
    Mat img_matches;

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
    
    if (obj.size() > 0) 
    {    
        //-- Get the corners from the image_1 ( the object to be "detected" )
        int max_x = 0;
        int min_x = 5000;//works as infinity for such image sizes
        int max_y = 0;
        int min_y = 5000;//works as infinity for such image sizes
        int curX, curY;
        for (size_t i = 0; i < keypoints1.size(); i++)
        {
            curX = keypoints1[i].pt.x;
            curY = keypoints1[i].pt.y;
            if (curX > max_x)
                max_x = curX;
            if (curY > max_y)
                max_y = curY;

            if (curX < min_x)
                min_x = curX;
            if (curY < min_y)
                min_y = curY;
        }

        obj_corners[0] = Point(min_x,min_y);
        obj_corners[1] = Point( max_x, min_y );
        obj_corners[2] = Point( max_x, max_y );
        obj_corners[3] = Point( min_x, max_y );

        //obj_corners[0] = Point(0,0);
        //obj_corners[1] = Point( img1.cols, 0 );
        //obj_corners[2] = Point( img1.cols, img1.rows );
        //obj_corners[3] = Point( 0, img1.rows );

        Mat H = findHomography( obj, scene, RANSAC, 3 );

        //std::cout << H << std::endl;
        

        if (countNonZero(H) < 1) 
        {
            std::cout << outputCounter++<< ": Not a proper match. " << 0 << std::endl;
        }
        else
        {
            perspectiveTransform( obj_corners, scene_corners, H);

            scene_corners_ = scene_corners;

            Mat drawing = Mat::zeros( img2.size(), img2.type() );

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

            for( size_t i = 0; i < good_matches.size(); i++ )
            {
                val = pointPolygonTest( contours[0], keypoints2[ good_matches[i].trainIdx ].pt , false );
                if (val >= 0)
                {
                    selected_matches.push_back(good_matches[i]);
                }
                
            }

            if (selected_matches.size() > 0)
            {
                std::cout << outputCounter++<< ": A proper match. " << selected_matches.size()  << std::endl;               
            }
            else
            {
                std::cout << outputCounter++<< ": Not a proper match. " << 0  << std::endl;
            }
            
        }
    }
    else
    {
        std::cout << outputCounter++<< ": Not a proper match. " << 0  << std::endl;
    }

    drawMatches( img1, keypoints1, img2, keypoints2,
                 selected_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
    
    /// Draw contours
    for( int i = 0; i< contours.size(); i++ )
    {
        drawContours( img_matches, contours, i, Scalar( 255, 0, 0), 2, 8, hierarchy, 0, Point2f( (float)img1.cols, 0) );
    }

    line( img_matches,
            scene_corners[0] + Point2f( (float)img1.cols, 0), scene_corners[1] + Point2f( (float)img1.cols, 0),
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
            scene_corners[1] + Point2f( (float)img1.cols, 0), scene_corners[2] + Point2f( (float)img1.cols, 0),
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
            scene_corners[2] + Point2f( (float)img1.cols, 0), scene_corners[3] + Point2f( (float)img1.cols, 0),
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( img_matches,
            scene_corners[3] + Point2f( (float)img1.cols, 0), scene_corners[0] + Point2f( (float)img1.cols, 0),
            Scalar( 0, 255, 0), 2, LINE_AA );

    return img_matches;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    const char* keys =
        "{ h help     |                  | print help message  }"
        "{ t test     | test.jpg          | specify left image  }"
        "{ o output   | SURF_output.jpg  | specify output save path }"
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

    UMat img1, srcImg;

    std::string outpath = cmd.get<std::string>("o");
    std::cout << "Outpath = " <<outpath<< std::endl;

    std::string testName = cmd.get<std::string>("t");
    std::cout << "Test = " <<testName<< std::endl;

    imread(testName, CV_LOAD_IMAGE_GRAYSCALE).copyTo(srcImg);
    if(srcImg.empty())
    {
        std::cout << "Couldn't load " << testName << std::endl;
        cmd.printMessage();
        return EXIT_FAILURE;
    }

    //crop scrImg into img1
    // Setup a rectangle to define your region of interest
    int imgHeight = srcImg.size().height;
    int imgWidth = srcImg.size().width;
    //the following values should come from a region detection algorithm background susbtracion using white parts
    int start_X = imgWidth/3;
    int start_Y = imgHeight/4;
    int width_X = imgWidth/3;
    int width_Y = imgHeight/2;
    Rect myROI(start_X, start_Y, width_X, width_Y);

    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    img1 = srcImg(myROI);
    //srcImg.copyTo(img1);

    //start loop

    //String folderpath = "/Users/Hassan/Workspace/OpenCV/Dataset/BigBird";
    String folderpath = "/Users/Hassan/Workspace/OpenCV/Dataset/BigBird";

    std::vector<String> filenames;
    cv::glob(folderpath, filenames);

    std::cout <<filenames.size()<< std::endl;
    for (size_t i=0; i<filenames.size(); i++)
    {
        UMat img2;

        imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE).copyTo(img2);
        if(img2.empty())
        {
            std::cout << "Couldn't load " << filenames[i] << std::endl;
            cmd.printMessage();
            return EXIT_FAILURE;
        }

        //declare input/output
        std::vector<KeyPoint> keypoints1, keypoints2;
        std::vector< std::vector<DMatch> > matches;
        std::vector<DMatch> backward_matches;
        //std::vector<DMatch> matches;

        //UMat _descriptors1, _descriptors2;
        /*
        Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
            descriptors2 = _descriptors2.getMat(ACCESS_RW);
        */
        Mat descriptors1, descriptors2;

        //instantiate detectors/matchers
        SURFDetector surf;
        
        //SIFTDetector sift;
        
        //el 3 lines el gayeen habal
        //detector.create();//create (int nfeatures=0, int nOctaveLayers=3, double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)
        //Mat mask = Mat::zeros(img1.size(), CV_8UC1);  //NOTE: using the type explicitly
        
        //SURFMatcher<BFMatcher> matcher;
        BFMatcher matcher;

        surf(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
        surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
        //sift(img1.getMat(ACCESS_READ), Mat(), keypoints1, descriptors1);
        //sift(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
        
        //matcher.match(descriptors1, descriptors2, matches);
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);// Find two nearest matches
        matcher.match(descriptors2, descriptors1, backward_matches);
        

        //std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
        //std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;


        std::vector<Point2f> corner;
        Mat img_matches = findGoodMatches(img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), keypoints1, keypoints2, matches, backward_matches, corner);

        //-- Show detected matches

        //std::cout << "after2" << std::endl << std::endl;
        while(img_matches.empty()){};
        namedWindow("surf matches", 0);
        imshow("surf matches", img_matches);
        //imwrite(outpath, img_matches);

        char c = (char)waitKey(0);
        if(c == 's')
            i += 100;
    }

    //end loop
    return EXIT_SUCCESS;
}
