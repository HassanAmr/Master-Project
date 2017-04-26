#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

using namespace cv;

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
int main(int argc, char* argv[])
{
    

    //read the matrices in a file
    FileStorage matrices("output/Matrices.xml", FileStorage::READ);
    Mat currM;
    std::string filename;
    while (1)
    {
        std::cout<<std::endl<<"Please enter the file name: ";
        std::cin>>filename;
//        std::cout<<std::endl;

        matrices[filename] >> currM;

        //check if col size is more than 3
        if (currM.cols > 3)
        {
            Mat H = Mat(currM,Rect(0,0,3,currM.rows));
            Mat F = Mat(currM,Rect(3,0,3,currM.rows));
            Mat E = Mat(currM,Rect(6,0,3,currM.rows));
            Mat R = Mat(currM,Rect(9,0,6,currM.rows));
            Mat T = Mat(currM,Rect(15,0,2,currM.rows));
            std::cout << std::endl
            << "H = " << std::endl
            <<  H << std::endl << std::endl
            << "F = " << std::endl
            <<  F << std::endl << std::endl
            << "E = " << std::endl
            <<  E << std::endl << std::endl
            << "R = " << std::endl
            <<  R << std::endl << std::endl
            << "T = " << std::endl
            <<  T << std::endl << std::endl << std::endl;
        }
        else
        {
            std::cout<<std::endl<<"Not a valid matrix" <<std::endl;
        }
    }

    //descriports and matrices are not needed anymore after this point
    matrices.release();
    return EXIT_SUCCESS;
}
