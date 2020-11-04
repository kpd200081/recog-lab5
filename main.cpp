#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

#define CUTED_SIZE 640

static int x_offset(490), y_offset(0), w1(855), w2(220), h(175), w_size_t(32), w_size(64),
    treshold(200);

void wtrb_pos(int pos, void*) {
    if(pos < 1) {
        w_size_t = 1;
        setTrackbarPos("window size", "Main", 1);
    }
    if(CUTED_SIZE % w_size_t == 0) {
        w_size = w_size_t;
    }
    setTrackbarPos("window size", "Main", w_size);
}

int main() {
    VideoCapture cap("../solidWhiteRight.mp4");

    Mat frame, res, transform, hls, dtc, dtc_show, window;

    while(frame.empty()) {
        cap >> frame;
    }

    namedWindow("Main", WINDOW_AUTOSIZE);
    namedWindow("Cutted", WINDOW_AUTOSIZE);
    namedWindow("Detected", WINDOW_AUTOSIZE);
    createTrackbar("x offset", "Main", &x_offset, frame.size[1]);
    createTrackbar("y offset", "Main", &y_offset, frame.size[0]);
    createTrackbar("down width", "Main", &w1, frame.size[1]);
    createTrackbar("up width", "Main", &w2, frame.size[1]);
    createTrackbar("height", "Main", &h, frame.size[0]);
    createTrackbar("window size", "Main", &w_size_t, CUTED_SIZE / 4, &wtrb_pos);
    createTrackbar("white treshold in window", "Main", &treshold, 3000);

    std::vector<std::vector<Point>> pols;
    std::vector<Point> pol;
    std::vector<Point2f> persp_dst, persp_src;
    persp_dst.push_back(Point2f(0, 0));
    persp_dst.push_back(Point2f(CUTED_SIZE, 0));
    persp_dst.push_back(Point2f(CUTED_SIZE, CUTED_SIZE));
    persp_dst.push_back(Point2f(0, CUTED_SIZE));
    std::vector<Point2f> src_p;
    std::vector<Point2f> dtc_p;
    while(true) {
        cap >> frame;

        if(frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        frame.copyTo(res);

        pol.clear();
        pols.clear();
        pol.push_back(Point(x_offset - w1 / 2, frame.size[0] - y_offset));
        pol.push_back(Point(x_offset + w1 / 2, frame.size[0] - y_offset));
        pol.push_back(Point(x_offset + w2 / 2, frame.size[0] - y_offset - h));
        pol.push_back(Point(x_offset - w2 / 2, frame.size[0] - y_offset - h));
        pols.push_back(pol);
        polylines(frame, pols, true, Scalar(0, 255, 0), 2);

        putText(
            frame,
            format("FPS: %3.2f", cap.get(CAP_PROP_FPS)),
            Point(10, 25),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        putText(
            frame,
            format("Frame:        %5.0f", cap.get(CAP_PROP_POS_FRAMES)),
            Point(10, 50),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        putText(
            frame,
            format("Frames total: %5.0f", cap.get(CAP_PROP_FRAME_COUNT)),
            Point(10, 75),
            FONT_HERSHEY_TRIPLEX,
            0.75,
            Scalar(0, 255, 0));

        persp_src.clear();
        persp_src.push_back(Point2f(x_offset - w2 / 2, frame.size[0] - y_offset - h));
        persp_src.push_back(Point2f(x_offset + w2 / 2, frame.size[0] - y_offset - h));
        persp_src.push_back(Point2f(x_offset + w1 / 2, frame.size[0] - y_offset));
        persp_src.push_back(Point2f(x_offset - w1 / 2, frame.size[0] - y_offset));
        transform = getPerspectiveTransform(persp_src, persp_dst);
        warpPerspective(res, res, transform, { int(persp_dst[2].x), int(persp_dst[2].y) });

        imshow("Cutted", res);

        cvtColor(res, hls, COLOR_BGR2HLS);
        inRange(hls, Scalar(0, 0, 0), Scalar(255, 215, 255), dtc);
        bitwise_not(dtc, dtc);
        src_p.clear();
        dtc_p.clear();
        dtc.copyTo(dtc_show);
        for(int i = 0; i < CUTED_SIZE / w_size; i++) {
            for(int j = 0; j < 2 * CUTED_SIZE / w_size - 1; j++) {
                Rect roi(j * w_size / 2, i * w_size, w_size, w_size);
                window = dtc(roi);
                Moments ms = moments(window, true);
                if(ms.m00 > treshold) {
                    Point2f p(
                        j * w_size / 2 + float(ms.m10 / ms.m00),
                        i* w_size + float(ms.m01 / ms.m00));
                    bool exist = false;
                    for(size_t k = 0; k < dtc_p.size(); k++) {
                        if(norm(dtc_p[k] - p) < 10) {
                            exist = true;
                        }
                    }
                    if(!exist) {
                        dtc_p.push_back(p);
                        circle(dtc_show, p, 5, Scalar(128), -1);
                    }
                    rectangle(dtc_show, roi, Scalar(255), 2);
                }
            }
        }

        imshow("Detected", dtc_show);

        if(dtc_p.size() > 0) {
            perspectiveTransform(dtc_p, src_p, transform.inv());
        }
        for(size_t i = 0; i < dtc_p.size(); i++) {
            circle(frame, src_p[i], 5, Scalar(0, 0, 255), -1);
        }

        imshow("Main", frame);

        if(waitKey(10) == 27)
            break; // ESC
    }

    cap.release();
    return 0;
}
