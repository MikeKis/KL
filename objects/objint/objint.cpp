#include <chrono>
#include <iostream>
#include <thread>
#include <string>

#include <zmq.hpp>

#include "objint.h"

using namespace std;
using namespace cv;
using namespace chrono;

int framesTrained = 0;

int receiveInt(zmq::socket_t &socket)
{
    zmq::message_t message;
    socket.recv(&message);
    string data(static_cast<char *>(message.data()), message.size());
    return stoi(data);
}

void trainOnRoi(Mat &receivedRoi, int classId)
{
    cv::Mat resizedRoi;  // frame for training (31 x 31)
    Preprocess(receivedRoi, resizedRoi);
    Learn(resizedRoi, classId);
    putText(receivedRoi, to_string(framesTrained), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
}

int predictOnRoi(Mat &receivedRoi)
{
    cv::Mat resizedRoi; // frame for inference (31 x 31)
    cv::resize(receivedRoi, resizedRoi, cv::Size(31, 31));

    // Put spike neural network inference code here
    // ...

    int predictedClass = rand() % 10;  // Replace predictedClass variable with actual prediction

    cout << "Predicted class: " << predictedClass << endl;
    return predictedClass;
}

string processFrame(int boxCount, int roiHeight, int trainingMode, int classId, const Mat &receivedImage)
{
    string predictedClasses = "";

    for (int boxIndex = 0; boxIndex < boxCount; ++boxIndex)
    {
        int x1 = boxIndex * roiHeight;
        int x2 = x1 + roiHeight;
        Mat receivedRoi = receivedImage(Range(0, roiHeight), Range(x1, x2));

        if (trainingMode == 1)
        {
            trainOnRoi(receivedRoi, classId);
            framesTrained++;
            imshow("Training frame (C++)", receivedRoi);
            waitKey(100);  // change to 1 ms after the training implementation
        }
        else
        {
            framesTrained = 0;
            int predictedClass = predictOnRoi(receivedRoi);
            predictedClasses.append(to_string(predictedClass) + ",");
            imshow("Inference frame (C++)", receivedRoi);
            waitKey(1);
        }
    }
    return predictedClasses;
}

void sendResults(zmq::socket_t &push_socket, const string &predictedClasses, long processingTime, long totalLatency)
{
    string performance = to_string(processingTime);
    string latency = to_string(totalLatency);
    push_socket.send(predictedClasses.c_str(), predictedClasses.size(), ZMQ_NOBLOCK);
    push_socket.send(latency.c_str(), latency.size(), ZMQ_NOBLOCK);
    push_socket.send(performance.c_str(), performance.size(), ZMQ_NOBLOCK);
}

int main()
{
    zmq::context_t context(2);
    zmq::socket_t pull_socket(context, ZMQ_PULL);
    zmq::socket_t push_socket(context, ZMQ_PUSH);

    pull_socket.connect("tcp://127.0.0.1:8001");
    push_socket.connect("tcp://127.0.0.1:8002");

    for (;;)
    {
        auto startTimeLoop = high_resolution_clock::now();
        int boxCount = receiveInt(pull_socket);
        int roiHeight = receiveInt(pull_socket);
        int classId = receiveInt(pull_socket);
        int trainingMode = receiveInt(pull_socket);

        cout << "Received box count: " << boxCount << endl;
        cout << "Received roi height: " << roiHeight << endl;
        cout << "Received class id for training: " << classId << endl;
        cout << "Received training mode: " << trainingMode << endl;

        int frameWidth = roiHeight * boxCount;
        zmq::message_t imageMessage;
        pull_socket.recv(&imageMessage);
        Mat receivedImage(roiHeight, frameWidth, CV_8U, imageMessage.data());

        auto startTimeProcessing = high_resolution_clock::now();
        string predictedClasses = processFrame(boxCount, roiHeight, trainingMode, classId, receivedImage);
        auto stopTime = high_resolution_clock::now();

        auto processingTime = duration_cast<milliseconds>(stopTime - startTimeProcessing).count();
        auto totalLatency = duration_cast<milliseconds>(stopTime - startTimeLoop).count();

        cout << "All predicted classes to send: " << predictedClasses << endl;
        cout << "Total latency: " << totalLatency << endl;
        cout << "Processing time: " << processingTime << endl;

        sendResults(push_socket, predictedClasses, processingTime, totalLatency);
    }
    return 0;
}
