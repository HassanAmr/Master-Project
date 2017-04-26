#include <iostream>
#include <stdio.h>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>

using namespace cv;
using namespace cv::xfeatures2d;


int outputCounter = 1;
const float ratio_Lowe = 0.8f; // As in Lowe's paper; can be tuned

//Always make sure this is accounted for

const int X_RES = 640;
const int Y_RES = 512;

const int GOOD_PORTION = 10;

//junk delete later
FileStorage f_verification("output/verification/F_verification.xml", FileStorage::WRITE);
FileStorage e_verification("output/verification/E_verification.xml", FileStorage::WRITE);

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


    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }



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

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(min_x,min_y);
    obj_corners[1] = Point( max_x, min_y );
    obj_corners[2] = Point( max_x, max_y );
    obj_corners[3] = Point( min_x, max_y );
    std::vector<Point2f> scene_corners(4);


    //obj_corners[0] = Point(0,0);
    //obj_corners[1] = Point( cols, 0 );
    //obj_corners[2] = Point( cols, rows );
    //obj_corners[3] = Point( 0, rows );

    Mat H = findHomography( obj, scene, RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
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

//later should be void again, and Mat should go to the new function that will be specifically responsible for homography
Mat findGoodMatches(
    int cols, int rows,//the columns and rows that cover exactly how big the object is, used for RANSAC homography
    const std::vector<KeyPoint>& keypoints1,
    const std::vector<KeyPoint>& keypoints2,
    std::vector< std::vector<DMatch> >& matches,
    std::vector<DMatch>& backward_matches,
    std::vector<DMatch>& selected_matches,
    std::string currImgText
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
    //std::cout<<good_matches.size() << "    ";
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
    Mat H, M; //The matrices to be returned
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
        //obj_corners[1] = Point( cols, 0 );
        //obj_corners[2] = Point( cols, rows );
        //obj_corners[3] = Point( 0, rows );

        H = findHomography( obj, scene, RANSAC, 3 );

        //std::cout << H << ": Not a proper match. " << good_matches.size() << std::endl;

        if (countNonZero(H) < 1)
        {
            //std::cout << outputCounter++<< ": Not a proper match. " << selected_matches.size() << std::endl;
        }
        else
        {
            perspectiveTransform( obj_corners, scene_corners, H);

            //find out later what this does
            //scene_corners_ = scene_corners;

            //Mat drawing = Mat::zeros( img2.size(), img2.type() );
            //using searchImg since img2 is currently not available, and both are the same size.
            //later should be set to the region where the object surely is.
            Mat drawing = Mat::zeros( X_RES, Y_RES, CV_8UC1);

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
            
            if (selected_matches.size() >= 8)
            {
                //-- Get the points corresponding to the selected matches
                std::vector<Point2f> queryPoints;
                std::vector<Point2f> refPoints;

                for( int i = 0; i < selected_matches.size(); i++ )
                {
                    //-- Get the keypoints from the good matches
                    queryPoints.push_back( keypoints1[ selected_matches[i].queryIdx ].pt );
                    refPoints.push_back( keypoints2[ selected_matches[i].trainIdx ].pt );
                }

                Mat F = findFundamentalMat(queryPoints, refPoints, CV_FM_8POINT);

                std::vector<double> verifyValues;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    /* code */
                    Mat queryMatrix(queryPoints[i]);
                    Mat refMatrix(refPoints[i]);
                    queryMatrix.convertTo(queryMatrix,DataType<double>::type);
                    refMatrix.convertTo(refMatrix,DataType<double>::type);
                    Mat one = cv::Mat::ones(1, 1, DataType<double>::type);
                    queryMatrix.push_back(one);
                    refMatrix.push_back (one);

                    Mat final = queryMatrix.t() * F * refMatrix;
                    verifyValues.push_back(final.at<double>(0,0));
                }

                bool ok = false;
                float acc_deviation = 0;
                int cnt = 0;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    acc_deviation += fabs(verifyValues[i]);
                    if (fabs(verifyValues[i]) < 2)
                    {
                        ok = true;
                        cnt ++;
                    }
                }

                float u_deviation = acc_deviation/selected_matches.size();
                if (ok)
                {
                    f_verification << currImgText<< ("Yes: " + std::to_string(cnt) + " out of " + std::to_string(selected_matches.size()) + " selected matches with a mean deviation of " + std::to_string(u_deviation) + " pixels");
                }
                else
                {
                    f_verification << currImgText<< "No";
                }

                Mat K1 = (Mat_<double>(3,3) << 1076.8879, 0.0, 312.05695, 0.0, 1076.8904, 244.55385, 0.0, 0.0, 1.0);
                //Mat K2 = (Mat_<double>(3,3) << 538.44395, 0.0, 312.05695, 0.0, 538.4452, 244.55385, 0.0, 0.0, 1.0);
                //Mat K1 = (Mat_<double>(3,3) << 538.44395, 0.0, 312.05695, 0.0, 538.4452, 244.55385, 0.0, 0.0, 0.5);
                Mat W = (Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

                Mat E = K1.t()* F * K1;
                //K1 E = K1.t()* F * K2;
                //Mat E = K2.t()* F * K1;

                verifyValues.clear();
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    /* code */
                    Mat queryMatrix(queryPoints[i]);
                    Mat refMatrix(refPoints[i]);
                    queryMatrix.convertTo(queryMatrix,DataType<double>::type);
                    refMatrix.convertTo(refMatrix,DataType<double>::type);
                    Mat one = cv::Mat::ones(1, 1, DataType<double>::type);
                    queryMatrix.push_back(one);
                    refMatrix.push_back (one);

                    Mat final = queryMatrix.t() * E * refMatrix;
                    verifyValues.push_back(final.at<double>(0,0));
                }

                ok = false;
                acc_deviation = 0;
                cnt = 0;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    acc_deviation += fabs(verifyValues[i]);
                    if (fabs(verifyValues[i]) < 2)
                    {
                        ok = true;
                        cnt ++;
                    }
                }

                u_deviation = acc_deviation/selected_matches.size();
                if (ok)
                {
                    e_verification << currImgText<< ("Yes: " + std::to_string(cnt) + " out of " + std::to_string(selected_matches.size()) + " selected matches with a mean deviation of " + std::to_string(u_deviation) + " pixels");
                }
                else
                {
                    e_verification << currImgText<< "No";
                }

                SVD svd(E);
                Mat R1 = svd.u *W * svd.vt;
                Mat R2 = svd.u *W.t() * svd.vt;

                //std::cout<< svd.u << std::endl<< svd.w <<std::endl<< svd.u.t()<<std::endl;
                //Mat L = Mat::zeros(3, 3, CV_64F);
                //L.at<double>(0,0) = svd.w.at<double>(0);
                //L.at<double>(1,1) = svd.w.at<double>(1);
                //L.at<double>(2,2) = svd.w.at<double>(2);
                Mat u3(3, 1, DataType<double>::type);
                u3.at<double>(0,0) = svd.u.at<double>(0,2);
                u3.at<double>(1,0) = svd.u.at<double>(1,2);
                u3.at<double>(2,0) = svd.u.at<double>(2,2);
                double min, max;
                minMaxLoc(u3, &min, &max);
                max = std::max(fabs(min),max);
                //std::cout<<svd.u<<std::endl;
                //std::cout<<u3<<std::endl;
                Mat T1(3, 1, DataType<double>::type);
                Mat T2(3, 1, DataType<double>::type);
                T1.at<double>(0,0) = u3.at<double>(0,0)/max;
                T1.at<double>(1,0) = u3.at<double>(1,0)/max;
                T1.at<double>(2,0) = u3.at<double>(2,0)/max;

                T2.at<double>(0,0) = -1 * (u3.at<double>(0,0)/max);
                T2.at<double>(1,0) = -1 * (u3.at<double>(1,0)/max);
                T2.at<double>(2,0) = -1 * (u3.at<double>(2,0)/max);

                //std::cout << L<<std::endl;
                //Mat T = svd.u *L * W * svd.u.t();
                Mat T;
                hconcat(T1,T2, T);
                Mat R;
                hconcat(R1, R2, R);
                Mat transformations;
                hconcat(R, T, transformations);
                Mat FE;
                hconcat(F, E, FE);
                //hconcat(H, T, M);
                Mat misc;
                hconcat(H, FE, misc);
                hconcat(misc, transformations, M);
                return M;
            }
        }
    }

    f_verification << currImgText<< "No";
    e_verification << currImgText<< "No";
    return H;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    /*
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
    */

    String queryName, backName, db_location, dataset_location;//  "/media/hassan/myPassport/Dataset/BigBIRD/";
    const String dataset_type = ".jpg"; //TODO: maybe set it as argument later
    //String db_location = "/media/hassan/myPassport/Dataset/db/";
    if (argc > 4)
    {

        backName = argv[1];
        queryName = argv[2];
        db_location = argv[3];
        dataset_location = argv[4];
        if (argc > 5)
        {
            printf("Too many arguments.\n\nPlease enter:\n\t1. The location of the background image\n\t2. The location of the query image\n\t2. The path of the db\n\t3. The location of the image files\n\n");
            return 1;
        }
    }
    else
    {
      printf("Too few arguments.\n\nPlease enter:\n\t1. The location of the background image\n\t2. The location of the query image\n\t2. The path of the db\n\t3. The location of the image files\n\n");
      return 1;
    }

    UMat img1, queryImg,backImg;
    Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
    //Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
    Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
    //create Background Subtractor objects
    //pMOG = bgsegm::createBackgroundSubtractorMOG(); //MOG approach
    pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach


    //std::string testName = cmd.get<std::string>("t");
    //std::cout << "Test = " <<testName<< std::endl;

    //imread(backName, CV_LOAD_IMAGE_GRAYSCALE).copyTo(backImg);
    resize(imread(backName, CV_LOAD_IMAGE_GRAYSCALE), backImg, Size(640, 512), 0, 0, INTER_AREA); 
    if(backImg.empty())
    {
        std::cout << "Couldn't load " << backName << std::endl;
        //cmd.printMessage();
        printf("Wrong input arguments.\n\nPlease enter:\n\t1. The location of the background image\n\t2. The location of the query image\n\t2. The path of the db\n\t3. The location of the image files\n\n");
        return EXIT_FAILURE;
    }
    //imread(queryName, CV_LOAD_IMAGE_GRAYSCALE).copyTo(queryImg);
    resize(imread(queryName, CV_LOAD_IMAGE_GRAYSCALE), queryImg, Size(640, 512), 0, 0, INTER_AREA); 
    if(queryImg.empty())
    {
        std::cout << "Couldn't load " << queryName << std::endl;
        //cmd.printMessage();
        printf("Wrong input arguments.\n\nPlease enter:\n\t1. The location of the background image\n\t2. The location of the query image\n\t2. The path of the db\n\t3. The location of the image files\n\n");
        return EXIT_FAILURE;
    }

    pMOG2->apply(backImg, fgMaskMOG2);
    pMOG2->apply(queryImg, fgMaskMOG2);
    //const Mat mask = fgMaskMOG2;

    imwrite("output/bg_subtraction.jpg", fgMaskMOG2);

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
    //queryImg.copyTo(img1);
    //resize(queryImg, img1, Size(640, 512), 0, 0, INTER_AREA); 
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

    String dscspath = db_location + "Desciptors.xml";
    String kptspath = db_location + "KeyPoints.xml";
    String idspath = db_location + "Mapping_IDs.xml";

    FileStorage dscs(dscspath, cv::FileStorage::READ);
    FileStorage kpts(kptspath, cv::FileStorage::READ);
    FileStorage ids(idspath, cv::FileStorage::READ);

    //std::cout << "db_location:"<< std::endl<<dscspath << std::endl << kptspath << std::endl<<idspath<<std::endl;

    //std::cout << "files_location: " << std::endl << dataset_location <<std::endl;

    int matchesFound = 0;

    std::vector< std::vector<DMatch> > final_matches;

    std::vector<Mat> allMatrices;
    std::vector< std::vector<int> > ranked_IDs;
    int cols = img1.cols;
    int rows = img1.rows;


    //Fetching data for the first iteration
    int index = 1;
    String curr_img;
    String indexValue = std::to_string(index);
    String filename = "node_" + indexValue;
    ids[filename] >> curr_img;
    kpts[filename] >> keypoints2;
    dscs[filename] >> descriptors2;
        //for (int i = 1; i <= 5400; i++)

    //store the matrices in a file
    FileStorage matrices("output/Matrices.xml", FileStorage::WRITE);


    while (curr_img != "")
    {
        //load descriptors2
        //surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);

        //std::cout << curr_img << " -> ";
        //std::vector<DMatch> matches;
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);// Find two nearest matches
        matcher.match(descriptors2, descriptors1, backward_matches);

        std::vector<DMatch> selected_matches;

        allMatrices.push_back( findGoodMatches(cols, rows, keypoints1, keypoints2, matches, backward_matches, selected_matches,curr_img.c_str()) );

        matrices << curr_img.c_str() << allMatrices.back();


        final_matches.push_back(selected_matches);
        matchesFound = selected_matches.size();

        //do a data structure for holding IDs with rank, rank being the number found just above.
        //push to this data structure
        //after this loop, this data structure should be sorted in descending order, and the selected amount should be filtered from it (top 10 for example).
        std::vector<int> currentItem;
        currentItem.push_back(index);
        currentItem.push_back(matchesFound); //rank
        ranked_IDs.push_back(currentItem);

        matches.clear();
        backward_matches.clear();
        selected_matches.clear();
        keypoints2.clear();
        descriptors2.release();


        //Fetching data for the next iteration
        index++;
        curr_img = "";
        String indexValue = std::to_string(index);
        String filename = "node_" + indexValue;
        ids[filename] >> curr_img;
        kpts[filename] >> keypoints2;
        dscs[filename] >> descriptors2;
    }

    //descriports and matrices are not needed anymore after this point
    matrices.release();
    dscs.release();
    f_verification.release();
    e_verification.release();
    


    //start formating the output
    std::cout << std::endl;
    std::cout <<index - 1 << std::endl;

    //-- Sort matches and preserve top 10% matches
    //std::sort(matches.begin(), matches.end());
    std::sort(ranked_IDs.begin(), ranked_IDs.end(), FirstColumnOnlyCmp());
    int currID, currRank;
    std::vector<DMatch> currMatches;
    Mat currM;
    UMat img2;
    String input_file, output_file;
    double currFitnessScore = 0.0;
    for (int i = 0; i < GOOD_PORTION; i++)
    {
        currID = ranked_IDs[i][0];
        currRank = ranked_IDs[i][1];
        currMatches = final_matches[currID - 1];    //minus 1 because IDs start at 1 while index start at 0
        currM = allMatrices[currID - 1];              //minus 1 because IDs start at 1 while index start at 0

        cv::String idValue = std::to_string(currID);
        cv::String filename = "node_" + idValue;
        kpts[filename] >> keypoints2;
        ids[filename] >> curr_img;
        input_file = dataset_location + curr_img + dataset_type;
        std::cout<<input_file<<std::endl;
        imread(input_file, CV_LOAD_IMAGE_GRAYSCALE).copyTo(img2);//get corresponding image

        Mat H = Mat(currM,Rect(0,0,3,currM.rows));
        Mat F = Mat(currM,Rect(3,0,3,currM.rows));
        Mat E = Mat(currM,Rect(6,0,3,currM.rows));
        Mat R = Mat(currM,Rect(9,0,6,currM.rows));
        Mat T = Mat(currM,Rect(15,0,2,currM.rows));
        currFitnessScore = (double)currMatches.size()/ (double)keypoints2.size();
        std::cout << curr_img << " -> "<< currFitnessScore << std::endl
        << "H = " << std::endl
        <<  H << std::endl << std::endl
        << "F = " << std::endl
        <<  F << std::endl << std::endl
        << "E = " << std::endl
        <<  E << std::endl << std::endl
        << "R = " << std::endl
        <<  R << std::endl << std::endl
        << "T = " << std::endl
        <<  T << std::endl << std::endl;

        //write image to disk
        Mat img_matches = drawGoodMatches(keypoints1, keypoints2, img1.getMat(ACCESS_READ), img2.getMat(ACCESS_READ), currMatches);
        while(img_matches.empty()){};


        //TODO: compute transform matrix and print

        //maybe do this part here to penaltilize non horizontals in drawGoodMatches??
        //currFitnessScore = (double)currMatches.size()/ (double)keypoints2.size();

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
