#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <omp.h>


using namespace cv;



Mat src; Mat dst;
int markerBoxSize = 16;

String SplitFilename (const std::string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}

String RemoveFileExtension (const std::string& str)
{
  std::size_t lastindex = str.find_last_of("."); 
  return str.substr(0, lastindex);
}

int main( int argc, char** argv )
{
    String folderpath, outputPath;

    if (argc > 2)
    {
      folderpath = argv[1];
      outputPath = argv[2];
      /*
      if (folderpath[folderpath.size()-1] == '/')
      {
        searchPath = folderpath.substr(0, folderpath.size()-1);
      }
      sampleName = SplitFilename(searchPath);
      std::cout<<searchPath<<std::endl;
      std::cout<<sampleName<<std::endl;
      */
      if (argc > 3)
      {
        printf("Too many arguments. Please enter the path of the files you wish to sample, then followed by the location in which you want to have the new samples.");
        return 1;
      }

    }
    else
    {
      printf("Too few arguments. Please enter the path of the files you wish to sample, then followed by the location in which you want to have the new samples.");
      return 1;
    }
    //String folderpath = "/Users/Hassan/Workspace/OpenCV/Dataset_BIG/copy_from_here/canon_ack_e10_box";
    //String outputPath = "/Users/Hassan/Workspace/OpenCV/Dataset/BigBIRD/";
    //String sampleName = "cam";
    std::vector<String> filenames;
    cv::glob(folderpath, filenames);

    std::cout <<filenames.size()<< std::endl;
    //int count = 1;
    //int j = 0;


    /*
    int iam = 0, np = 1;

    #pragma omp parallel default(shared) private(iam, np)
    {
      #if defined (_OPENMP)
        np = omp_get_num_threads();
        iam = omp_get_thread_num();
      #endif
      printf("Hello from thread %d out of %d\n", iam, np);
    }
    */

    std::vector<Point2f> corners, marker; //this will be filled by the detected corners
    bool patternfound = false;
    int l,k;
    //#pragma omp parallel default(shared) private(l, k)
    
    cv::String hMatrices_path = outputPath + "matrices/H_Matrices.xml";
    std::cout <<hMatrices_path<< std::endl;

    FileStorage hMatrices(hMatrices_path, FileStorage::WRITE);
    //#pragma omp parallel for private(patternfound, corners)
    Size patternsize;//this is just a dummy declaration because it is only declared inside a loop, and the compiler doesn't like it
    for (size_t i=0; i<filenames.size(); i++)
    {
        //if (j > 0){
        //    j++;
        //    if (j == 6)
        //        j = 0;
        //    continue;
        //}
        if (SplitFilename(filenames[i])[0] == '.')
        {
          continue;
        }


        Mat im = imread(filenames[i]);
        
        l = 7;
        while (!patternfound && l > 2)
        {
          k = 7;
          while (!patternfound && k > 2) 
          {
            patternsize.height = l;
            patternsize.width = k;
            patternfound = findChessboardCorners(im, patternsize, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
            k--;
          }
          l--;
        }

        if (!patternfound)
        {
          continue;
        }
        int x = 0;
        int y = 0;
        for (int  h = 0; h < patternsize.height; ++h)
        {
            for (int w = 0; w < patternsize.width; ++w)
            {
                Point2f p(w*markerBoxSize,h*markerBoxSize);
                marker.push_back(p);
            }
            /* code */
        }
        

        Mat dst;
        resize(im, dst, Size(640, 512), 0, 0, INTER_AREA); // resize to 640x512 resolution
        
        //std::cout<<patternsize<<std::endl;

        cv::String fileNameStr = SplitFilename(filenames[i]);
        std::ostringstream ss;
        ss <<outputPath << "images/"<< fileNameStr;
        String outputName = ss.str();

        std::cout <<outputName<< std::endl;

        imwrite(outputName , dst);

        drawChessboardCorners(dst, patternsize, corners, patternfound);
        drawChessboardCorners(dst, patternsize, marker, patternfound);
        patternfound = false;

        std::ostringstream ss2;
        ss2 <<outputPath << "annotation/"<< fileNameStr;
        
        String outputName2 = ss2.str();
        //std::cout <<patternsize.height<< std::endl;
        //std::cout <<patternsize.width<< std::endl;
        

        std::vector<Point2f> obj_corners(4);
        std::vector<Point2f> scene_corners(4);


        obj_corners[0] = Point(0,0);
        obj_corners[1] = Point( (patternsize.width - 1) * markerBoxSize, 0 );
        obj_corners[2] = Point( (patternsize.width - 1) * markerBoxSize, (patternsize.height - 1) * markerBoxSize );
        obj_corners[3] = Point( 0, (patternsize.height - 1) * markerBoxSize );

        Mat H = findHomography( marker, corners, RANSAC );
        perspectiveTransform( obj_corners, scene_corners, H);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( dst,
            scene_corners[0], scene_corners[1],
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( dst,
            scene_corners[1], scene_corners[2],
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( dst,
            scene_corners[2], scene_corners[3],
            Scalar( 0, 255, 0), 2, LINE_AA );
        line( dst,
            scene_corners[3], scene_corners[0],
            Scalar( 0, 255, 0), 2, LINE_AA );
        //std::cout <<scene_corners<< std::endl;
        imwrite(outputName2 , dst);

        //std::cout << RemoveFileExtension(SplitFilename(filenames[i])).c_str() << std::endl;

        hMatrices << RemoveFileExtension(fileNameStr).c_str() << H;
        corners.clear();
        marker.clear();
        obj_corners.clear();
        scene_corners.clear();
        H.release();
    }
    hMatrices.release();
    return 0;
}
