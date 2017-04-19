#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;



Mat src; Mat dst;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";

String SplitFilename (const std::string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}

int main( int argc, char** argv )
{
    String folderpath, outputPath, sampleName;

    if (argc > 2)
    {
      folderpath = argv[1];
      outputPath = argv[2];
      sampleName = SplitFilename(folderpath);
      if (argc > 3)
      {
        printf("Too much arguments. Please enter the path of the files you wish to sample, then followed by the location in which you want to have the new samples.");
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
    for (size_t i=0; i<filenames.size(); i++)
    {
        //if (j > 0){
        //    j++;
        //    if (j == 6)
        //        j = 0;
        //    continue;
        //}


        Mat im = imread(filenames[i]);
        Mat dst;
        resize(im, dst, Size(640, 512), 0, 0, INTER_AREA); // resize to 640x512 resolution
        std::ostringstream ss;
        ss <<outputPath << "/" <<sampleName<< "_"<<SplitFilename(filenames[i]);
        String outputName = ss.str();

        std::cout <<outputName<< std::endl;

        imwrite(outputName , dst);
        //count++;
        //j++;

    }
    return 0;
}
