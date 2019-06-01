/***********************************************************************************/
/*          OneCut - software for interactive image segmentation                   */
/*          "Grabcut in One Cut"                                                   */
/*          Meng Tang, Lena Gorelick, Olga Veksler, Yuri Boykov,                   */
/*          In IEEE International Conference on Computer Vision (ICCV), 2013       */
/*          https://github.com/meng-tang/OneCut                                    */
/*          Contact Author: Meng Tang (mtang73@uwo.ca)                             */
/***********************************************************************************/

#include "OneCut.h"
#include "myutil.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using namespace std;
int main(int argc, char * argv[])
{
	// set energy parameters and hyperparameters
	int ColorBinSize = 8; // size of color bin
	int GridConnectivity = 8; // 4, 8 or 16 connect Grid
	double WeightPotts = 9.0; // weight of Potts term
	MAXFLOW maxflowoption = IBFS; // either use BK or IBFS algorithm. BK is NOT recommended here.

    double total_time = 0.0;
	double accum_err = 0.0;
    for(int i = 0;i < 1000; ++i) {
        system("clear");
        printf("Processing image %d.jpg\n", 30001+i);
        outs("load input image");
        char image_name[100];
        sprintf(image_name, "../orderedImages/%d.bmp", 30001+i);
	    Table2D<RGB> image = loadImage<RGB>(image_name);
	    clock_t start = clock(); // Timing
    	OneCut onecut(image, ColorBinSize, GridConnectivity, maxflowoption); // 8 connect 32 bins per channel
	    onecut.print();
	
	    outs("load bounding box");
        char box_name[100];
        sprintf(box_name, "../orderedBoxes/%d_box.bmp", 30001 + i);
	    Table2D<int> box = loadImage<RGB>(box_name);
    	
	    onecut.constructbkgraph(box, WeightPotts);

	    outs("run maxflow/mincut");
    	Table2D<Label> segmentation = onecut.run();

	    outs("save segmentation");
        char result_name[100];
        sprintf(result_name, "./result/%d_res.bmp", 30001+i);
    	savebinarylabeling(image, segmentation, result_name);
     
        // timing
        double delta_t = (double)(clock()-start)/CLOCKS_PER_SEC;
    	cout<<"\nIt takes "<< delta_t <<" seconds!"<<endl;
        total_time += delta_t;

    	// segmentation error rate
        char mask_name[100];
        sprintf(mask_name, "../orderedTruths/%d_gt.bmp", 30001+i);
    	Table2D<int> groundtruth = loadImage<RGB>(mask_name); // ground truth
	    double errorrate = geterrorrate(segmentation, groundtruth, countintable(box, 0));
        accum_err += errorrate;
	    outv(errorrate);
    }
    cout << "Average error rate: " << accum_err / 1000 << endl;
    cout << "Average time for each image: " << total_time / 1000 << endl;

	return -1;
}
