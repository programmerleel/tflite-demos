
#ifndef _UTILS_H_
#define _UTILS_H_

#include <map>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#ifdef _WIN32
#include <windows.h>
// 显示文件路径或者文件名
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define LOGDT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
// 不显示文件路径或者文件名
//#define LOGDT(fmt, tag, ...)                                                                                           \
//    fprintf(stdout, ("D/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
//#define LOGIT(fmt, tag, ...)                                                                                           \
//    fprintf(stdout, ("I/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
//#define LOGET(fmt, tag, ...)                                                                                           \
//    fprintf(stderr, ("E/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif  //_WIN32


// >>> 重写LOG宏, 覆盖tnn/core/macro.h, __PRETTY_FUNCTION__带参数, __FUNCTION__不带参数
#ifdef __ANDROID__
#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "pthread.h"
// 显示文件路径或者文件名
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOGDT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_DEBUG, tag, ("%s [File %s][Line %d] " fmt), __FUNCTION__, __FILENAME__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("D/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOGIT(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_INFO, tag, ("%s [File %s][Line %d] " fmt), __FUNCTION__, __FILENAME__,          \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stdout, ("I/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define LOGET(fmt, tag, ...)                                                                                           \
    __android_log_print(ANDROID_LOG_ERROR, tag, ("%s [File %s][Line %d] " fmt), __FUNCTION__, __FILENAME__,         \
                        __LINE__, ##__VA_ARGS__);                                                                      \
    fprintf(stderr, ("E/%s: %s [File %s][Line %d] " fmt), tag, __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
// 不显示文件路径或者文件名
//#define LOGDT(fmt, tag, ...)                                                                                           \
//    __android_log_print(ANDROID_LOG_DEBUG, tag, ("%s [Line %d] " fmt), __FUNCTION__,         \
//                        __LINE__, ##__VA_ARGS__);                                                                      \
//    fprintf(stdout, ("D/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
//#define LOGIT(fmt, tag, ...)                                                                                           \
//    __android_log_print(ANDROID_LOG_INFO, tag, ("%s [Line %d] " fmt), __FUNCTION__,          \
//                        __LINE__, ##__VA_ARGS__);                                                                      \
//    fprintf(stdout, ("I/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
//#define LOGET(fmt, tag, ...)                                                                                           \
//    __android_log_print(ANDROID_LOG_ERROR, tag, ("%s [Line %d] " fmt), __FUNCTION__,         \
//                        __LINE__, ##__VA_ARGS__);                                                                      \
//    fprintf(stderr, ("E/%s: %s [Line %d] " fmt), tag, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif  //__ANDROID__

#ifdef PSPL_ENABLE_PRINT_LOG
#define LOGD(fmt, ...) if(enable_print_log()) LOGDT(fmt, "PSPLAI", ##__VA_ARGS__)
#define LOGI(fmt, ...) if(enable_print_log()) LOGIT(fmt, "PSPLAI", ##__VA_ARGS__)
#define LOGE(fmt, ...) if(enable_print_log()) LOGET(fmt, "PSPLAI", ##__VA_ARGS__)
#else
#define LOGD(fmt, ...)
#define LOGI(fmt, ...)
#define LOGE(fmt, ...)
#endif

namespace fs = std::filesystem;

using std::chrono::time_point;
using std::chrono::system_clock;

bool enable_print_log();

namespace common_util {
    void log_toggle(bool e);

    struct DetObj {
        float xmin = 0.0f;
        float ymin = 0.0f;
        float xmax = 0.0f;
        float ymax = 0.0f;
        float h = 0.0;
        float w = 0.0;
        float score = 0.0f;
        int class_idx = 0;
    };

    struct BoxObj {
        int xmin;
        int ymin;
        int xmax;
        int ymax;
        std::string category;
        BoxObj(int xmin_, int ymin_, int xmax_, int ymax_, std::string category_)
            :xmin(xmin_), ymin(ymin_), xmax(xmax_), ymax(ymax_), category(category_) {}
    };

    // 这个在linux win下通用，而win下的clock_t类型在linux下是ms*1000
    class SampleTimer {
    public:
        SampleTimer() {};
        void Start();
        void Stop();
        void Reset();
        double GetTime();

    private:
        time_point<system_clock> start_;
        time_point<system_clock> stop_;
    };

    class PsplMutex {
    public:
        PsplMutex() { Init(); };
        ~PsplMutex() { Release(); };
        void Lock();
        void UnLock();
    private:
        void Init();
        void Release();
        bool init_done_ = false;
#ifdef _WIN32
        CRITICAL_SECTION m_;
#elif __linux__
        pthread_mutex_t m_;
#else
#endif
    };

#ifdef _WIN32
    std::string utf8_to_ansi(const std::string& str);

    std::string ansi_to_utf8(const std::string& str);
#endif

    void nms(std::vector<DetObj>& inputs, std::vector<DetObj>& outputs, float nms_iou_thresh);

    void nms_2(std::vector<DetObj>& inputs, std::vector<DetObj>& outputs, float nms_iou_thresh, bool agnostic = true);

    void softnms(std::vector<DetObj>& inputs, std::vector<DetObj>& outputs, float sigma, float score_thresh);

    // 拼接路径
    std::string join_path(const std::string& path_a, const std::string& path_b);

    bool is_utf8(const std::string& str);

    // windows linux通用的sleep函数 https://stackoverflow.com/questions/10918206/cross-platform-sleep-function-for-c
    void sleep(int ms);

    std::string strip(const std::string& input);

    std::string replace_string(std::string input, const std::string& src, const std::string& dst);

    // 从字符串中分割解析数字
    std::vector<float> parse_numbers(std::string input, char dlm);

    std::vector<std::string> split_string(std::string input, std::string dlm);

    std::map<std::string, std::string> read_config(std::string file_path);

    std::map<std::string, std::string> read_config_from_buffer(char* buffer, size_t length);

    std::vector<std::vector<float>> read_ssdfpn_box_priors(std::string priors_path);

    std::vector<std::vector<float>> read_ssdfpn_box_priors_from_buffer(char* buffer, size_t length);

    int write_float_bytes(float* values, int n, std::string save_path);

    int write_float_bytes(std::vector<float> values, int n, std::string save_path);

    std::vector<float> read_float_bytes(std::string file_path);

    std::vector<float> read_float_bytes_from_buffer(char* buffer, size_t length);

    std::vector<std::vector<float>> unsqueeze_box_priors(std::vector<float> input, int rows);

    std::vector<float> squeeze_box_priors(std::vector<std::vector<float>> input);

    std::vector<std::vector<float>> read_ssdfpn_box_priors_from_bytes(std::string file_path);

    std::vector<std::vector<float>> read_ssdfpn_box_priors_from_bytes_buffer(char* buffer, size_t length);

    int read_filebytes(const char* file_path, char*& buffer, size_t& size);

    int opencv_read_bgr_from_filebytes(cv::Mat& bgr_rgb, char* buffer, int size);

    int opencv_read_rgb(cv::Mat& cv_rgb, std::string image_path);

    int opencv_efficientdet_resize_pad(const cv::Mat& src, cv::Mat& dst, int dst_h, int dst_w, float& scale);

    int opencv_efficientnet_crop_resize(const cv::Mat& src, cv::Mat& dst, int dst_h, int dst_w, int crop_padding = 32);

    void sync_file();

    int move_file(std::string old_path, std::string new_path);

    int copy_file(std::string old_path, std::string new_path);

    int delete_file(std::string file_path);

    bool filepath_exists(std::string file_path);
}

#endif