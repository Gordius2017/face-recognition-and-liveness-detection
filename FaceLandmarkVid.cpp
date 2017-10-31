///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltruaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltruaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltruaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltruaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

//c sockets
#include <stdio.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

#include <queue>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;
bool X_pos = false, X_neg = false, Y_pos = false, Y_neg = false, Z_pos = false, Z_neg = false;\
int i = 0;

void error(char *msg)
{
    perror(msg);
    exit(1);
}

int send_message(int port, char* h, char* path, char* mesaj_){
	int portno = port;
    char *host = h;

    struct hostent *server;
    struct sockaddr_in serv_addr;
    int sockfd, bytes, sent, received, total, message_size;
    char *message, response[4096];

    //if (argc < 5) { puts("Parameters: <host> <port> <method> <path> [<data> [<headers>]]"); exit(0); }

    /* How big is the message? */
    message_size=0;
    message_size+=strlen("%s %s HTTP/1.0\n");
    message_size+=strlen("POST");                         /* method         */
    message_size+=strlen(path);                         /* path           */
	message_size+=strlen("Content-Length: %d\n")+10; /* content length */
    message_size+=strlen("\n");                          /* blank line     */
    message_size+=strlen(mesaj_);                     /* body           */

    /* allocate space for the message */
    message=(char *)malloc(message_size);

    /* fill in the parameters */

        sprintf(message,"%s %s HTTP/1.0\n", "POST", path);
        sprintf(message+strlen(message),"Content-Length: %d\n",strlen(mesaj_));
        strcat(message,"\n");                                /* blank line     */
        strcat(message,mesaj_);                           /* body           */

    /* What are we going to send? */
    printf("Request:\n%s\n",message);

    /* create the socket */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) error("ERROR opening socket");

    /* lookup the ip address */
    server = gethostbyname(h);
    if (server == NULL) error("ERROR, no such host");

    /* fill in the structure */
    memset(&serv_addr,0,sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(portno);
    memcpy(&serv_addr.sin_addr.s_addr,server->h_addr,server->h_length);

    /* connect the socket */
    if (connect(sockfd,(struct sockaddr *)&serv_addr,sizeof(serv_addr)) < 0)
        error("ERROR connecting");

    /* send the request */
    total = message_size;
    sent = 0;
    do {
        bytes = write(sockfd,message+sent,total-sent);
        printf("%d",bytes);
        if (bytes < 0)
            error("ERROR writing message to socket");
        if (bytes == 0)
            break;
        sent+=bytes;
    } while (sent < total);

    /* receive the response */
    memset(response,0,sizeof(response));
    total = sizeof(response)-1;
    received = 0;
    do {
        bytes = read(sockfd,response+received,total-received);
        if (bytes < 0)
            error("ERROR reading response from socket");
        if (bytes == 0)
            break;
        received+=bytes;
    } while (received < total);

    if (received == total)
        error("ERROR storing complete response from socket");

    /* close the socket */
    close(sockfd);

    /* process response */
    printf("Response:\n%s\n",response);

    free(message);
    return 0;
}


// Visualising the results
cv::Vec6d visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	cv::Vec6d pose_estimate_to_draw;
	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
		//cout << pose_estimate_to_draw[3]<<" "<< pose_estimate_to_draw[4]<<" " <<pose_estimate_to_draw[5] <<endl;
		i++;
		if(pose_estimate_to_draw[3] < -0.3){
			X_neg = true;
		}else if (pose_estimate_to_draw[3] > 0.3){
			X_pos = true;
		}

		if(pose_estimate_to_draw[4] < -0.3){
			Y_neg = true;
		}else if(pose_estimate_to_draw[4] > 0.3){
			Y_pos = true;
		}

		if(pose_estimate_to_draw[5] < -0.3){
			Z_neg = true;
		}else if (pose_estimate_to_draw[5]  > 0.3){
			Z_pos = true;
		}

		if(X_neg && X_pos && Y_neg && Y_pos && Z_pos && Z_neg && (i < 500)){
			send_message(8080, "10.5.5.39", "http://10.5.5.39/notification/grantaccess", "Is real");
			X_neg = false;
			X_pos = false;
			Y_neg = false;
			Y_pos = false;
			Z_pos = false;
			Z_neg = false;
			i=0;
		}else if(i > 500){
			send_message(8080, "10.5.5.39", "http://10.5.5.39/notification/denyaccess", "Is not real");
			X_neg = false;
			X_pos = false;
			Y_neg = false;
			Y_pos = false;
			Z_pos = false;
			Z_neg = false;
			i=0;
		}



		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
		
		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);
	}
	return pose_estimate_to_draw;
}




bool isReal(vector<cv::Vec6d> lastHeadOrientations){

	return false;
}

int main (int argc, char **argv)
{ 
	//SERVER INIT
	int sockfd, newsockfd, portno;
	socklen_t clilen;
	struct sockaddr_in serv_addr, cli_addr;
	int n;

	if (argc < 2) 
	{
	 fprintf(stderr,"ERROR, no port provided SERVER\n");
	 exit(1);
	}

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) 
	error("ERROR opening socket SERVER");
	bzero((char *) &serv_addr, sizeof(serv_addr));
	portno = atoi(argv[1]);
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(portno);
	if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
	      error("ERROR on binding SERVER");
	listen(sockfd,5);
	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newsockfd < 0) 
	  error("ERROR on accept SERVER");

	//CLIENT INIT
	int sockfdc, portnoc, nc;
	struct sockaddr_in serv_addrc;
	struct hostent *serverc;
	portnoc = 8001;
	sockfdc = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfdc < 0) 
	error("ERROR opening CLIENT socket");
	serverc = gethostbyname("localhost");
	if (serverc == NULL) {
	fprintf(stderr,"ERROR, no such host CLIENT\n");
	exit(0);
	}
	bzero((char *) &serv_addrc, sizeof(serv_addrc));
	serv_addrc.sin_family = AF_INET;
	bcopy((char *)serverc->h_addr, 
	(char *)&serv_addrc.sin_addr.s_addr,
	serverc->h_length);
	serv_addrc.sin_port = htons(portnoc);
	if (connect(sockfdc,(struct sockaddr *)&serv_addrc,sizeof(serv_addrc)) < 0) 
	error("ERROR connecting CLIENT");


	char *sockData = new char[921600];

	vector<string> arguments = get_arguments(argc, argv);

	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);	
	det_parameters.track_gaze = true;
	float fx = 0, fy = 0, cx = 0, cy = 0;

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	
	std::vector<cv::Vec6d> lastHeadOrientations;


	while (1){
		int size = 0;
        int bytes = 0;
	 	n = read(newsockfd,(char*)&size,sizeof(int));
		size = (size & 0x000000FFU) << 24 | (size & 0x0000FF00U) << 8 | (size & 0x00FF0000U) >> 8 | (size & 0xFF000000U) >> 24;
		for(int i=0;i<size;i+=bytes){
			bytes = read(newsockfd,sockData+i,size-i);
			if(bytes==-1){
	                printf("error");
	                break;
	            }
		}

		//send_image(8080, "10.5.5.39", "http://10.5.5.39/api/capture/frame/", sockData, size);

		std::vector<char> data(sockData,sockData+size);
		if(data.size()==0)
		{
		printf("Error");return 0;
		}
		cv::Mat captured_image = cv::imdecode(data,CV_LOAD_IMAGE_COLOR);
		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}		
		//cv::imshow("hello",captured_image);
		//cv::waitKey(1000);


		cv::Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cv::cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
					
			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, clnf_model, det_parameters);
			
			// Visualising the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			double detection_certainty = clnf_model.detection_certainty;

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);//left eye
			cv::Point3f gazeDirection1(0, 0, -1);//right eye


			if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
			{
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection0, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(clnf_model, gazeDirection1, fx, fy, cx, cy, false);
				//cout << gazeDirection0 << endl;
				//cout << gazeDirection1 << endl;
			}

			lastHeadOrientations.push_back( visualise_tracking(captured_image, clnf_model, det_parameters, gazeDirection0, gazeDirection1, 0, fx, fy, cx, cy));
			if (lastHeadOrientations.size()>30){
				lastHeadOrientations.erase(lastHeadOrientations.begin());
			}

			
				
			
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press == 'r')
			{
				clnf_model.Reset();
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}


		nc = write(sockfdc,(char *)&size,sizeof(size));
		nc = write(sockfdc,sockData,size);
	}
	


	

	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to world or camera coordinates
	//bool u;
	//string output_codec;
	//LandmarkDetector::get_video_input_output_params(files, out_dummy, output_video_files, u, output_codec, arguments);
	
	// The modules that are being used for tracking
	

	// Grab camera parameters, if they are not defined (approximate values will be used)
	//float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	//LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	// bool cx_undefined = false;
	// bool fx_undefined = false;
	// if (cx == 0 || cy == 0)
	// {
	// 	cx_undefined = true;
	// }
	// if (fx == 0 || fy == 0)
	// {
	// 	fx_undefined = true;
	// }

	// If multiple video files are tracked, use this to indicate if we are done
	// bool done = false;	
	// int f_n = -1;
	
	

	// while(!done) // this is not a for loop as we might also be reading from a webcam
	// {
		
	// 	string current_file;

	// 	// We might specify multiple video files as arguments
	// 	if(files.size() > 0)
	// 	{
	// 		f_n++;			
	// 	    current_file = files[f_n];
	// 	}
	// 	else
	// 	{
	// 		// If we want to write out from webcam
	// 		f_n = 0;
	// 	}
		
	// 	// Do some grabbing
	// 	cv::VideoCapture video_capture;
	// 	if( current_file.size() > 0 )
	// 	{
	// 		if (!boost::filesystem::exists(current_file))
	// 		{
	// 			FATAL_STREAM("File does not exist");
	// 			return 1;
	// 		}

	// 		current_file = boost::filesystem::path(current_file).generic_string();

	// 		INFO_STREAM( "Attempting to read from file: " << current_file );
	// 		video_capture = cv::VideoCapture( current_file );
	// 	}
	// 	else
	// 	{
	// 		INFO_STREAM( "Attempting to capture from device: " << device );
	// 		video_capture = cv::VideoCapture( device );

	// 		// Read a first frame often empty in camera
	// 		cv::Mat captured_image;
	// 		video_capture >> captured_image;
	// 	}

	// 	if (!video_capture.isOpened())
	// 	{
	// 		FATAL_STREAM("Failed to open video source");
	// 		return 1;
	// 	}
	// 	else INFO_STREAM( "Device or file opened");

		
		//video_capture >> captured_image;		

		// // If optical centers are not defined just use center of image
		// if (cx_undefined)
		// {
		// 	cx = captured_image.cols / 2.0f;
		// 	cy = captured_image.rows / 2.0f;
		// }
		// // Use a rough guess-timate of focal length
		// if (fx_undefined)
		// {
		// 	fx = 500 * (captured_image.cols / 640.0);
		// 	fy = 500 * (captured_image.rows / 480.0);

		// 	fx = (fx + fy) / 2.0;
		// 	fy = fx;
		// }		
	
		// int frame_count = 0;
		
		// // saving the videos
		// cv::VideoWriter writerFace;
		// if (!output_video_files.empty())
		// {
		// 	try
 	// 		{
		// 		writerFace = cv::VideoWriter(output_video_files[f_n], CV_FOURCC(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), 30, captured_image.size(), true);
		// 	}
		// 	catch(cv::Exception e)
		// 	{
		// 		WARN_STREAM( "Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
		// 	}
		// }

		// Use for timestamping if using a webcam
		//int64 t_initial = cv::getTickCount();

		
		
		//frame_count = 0;

		// Reset the model, for the next video
		clnf_model.Reset();
		
		// break out of the loop if done with all the files (or using a webcam)
		

	return 0;
}
