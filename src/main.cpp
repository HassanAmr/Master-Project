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

std::string SplitFilename (String str)
{
  std::size_t found = str.find_last_of("/\\"); //this is generic for windows and unix
  return str.substr(found+1);
}

int main(int argc, char* argv[])
{
    //String folderpath = "/Users/Hassan/Workspace/OpenCV/Dataset/BigBIRD";
    String folderpath;

    if (argc > 1)
    {
      folderpath = argv[1];
      if (argc > 2)
      {
        printf("Too much arguments. Please enter only the path of the files you wish to store.");
        return 1;
      }

    }
    else
    {
      printf("Too few arguments. Please enter only the path of the files you wish to store.");
      return 1;
    }
    FileStorage dscs("Desciptors.xml", FileStorage::WRITE);
    FileStorage kpts("KeyPoints.xml", FileStorage::WRITE);
    FileStorage ids("Mapping_IDs.xml", FileStorage::WRITE);

    std::vector<String> filenames;
    cv::glob(folderpath, filenames);

    std::cout <<filenames.size()<< std::endl;
    for (size_t i=0; i<filenames.size(); i++)
    {
        UMat img1;
        imread(filenames[i], CV_LOAD_IMAGE_COLOR).copyTo(img1);
        if(img1.empty())
        {
            std::cout << "Couldn't load " << filenames[i] << std::endl;
            return EXIT_FAILURE;
        }

        //declare input/output
        std::vector<KeyPoint> mykpts;

        UMat _myDesciptors;
        Mat myDesciptors = _myDesciptors.getMat(ACCESS_RW);

        //instantiate detectors/matchers
        SURFDetector surf;

        surf(img1.getMat(ACCESS_READ), Mat(), mykpts, myDesciptors);

        String nodeName = SplitFilename(filenames[i]);
        nodeName = nodeName.substr(0, nodeName.size() - 4);
        //std::cout << nodeName << std::endl;
        //return 0;
        //std::cout << "FOUND " << mykpts.size() << " keypoints" << std::endl;
        std::ostringstream ss;
        ss<< i + 1;
        String iName = "node_";
        iName += ss.str();

        dscs << iName.c_str() << myDesciptors;
        kpts << iName.c_str() << mykpts;
        ids<< iName.c_str() << nodeName;

    }


    dscs.release();
    kpts.release();
    ids.release();

    return EXIT_SUCCESS;
}
