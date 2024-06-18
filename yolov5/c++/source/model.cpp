#include "model.h"

// public
Model::Model() {}

Model::~Model() {
    Release();
}

int Model::Detect(cv::Mat& rgb_image, float score_thresh, std::vector<common_util::DetObj>& objs){
    // 只针对处理单类别box
    if (!CheckReady()) {
        LOGE("model has not init yet\n");
        return 0;
    }
    if (rgb_image.empty()) {
        LOGE("input rgb_image is empty\n");
        return 0;
    }

    // 清空objs
    objs.clear();

    int ret = 0;
    // 计时器
    common_util::SampleTimer st;
    float time_proc, time_inf, time_post, time_norm, time_resize;
    int image_height = rgb_image.rows;
    int image_width = rgb_image.cols;
    float image_scale = 1.0;
    float* output_data = nullptr;

    // 1. preprocess, 注意, 是先normalize再resize再pad (是否先resize再normalize再pad更快? android上normalize 28ms->1ms)
    st.Start();
    ret = NormalizeImage(rgb_image);
    if (ret != 1) return 0;
    st.Stop();
    time_norm = st.GetTime();

    // 这里img_src可能已经做过 -mean /stddev
    st.Start();
    ret = common_util::opencv_efficientdet_resize_pad(rgb_image, rgb_image, model_input_h_, model_input_w_, image_scale);
    if (ret != 1) {
        LOGE("opencv_efficientnet_crop_resize failed\n");
        return 0;
    }
    st.Stop();
    time_resize = st.GetTime();
    time_proc = time_norm + time_resize;

    // 2. inference
    st.Start();
    ret = Inference(rgb_image, output_data);
    st.Stop();
    time_inf = st.GetTime();

    // 3. postprocess
    st.Start();
    DetectPostProcess(output_data, objs, score_thresh, image_height, image_width, image_scale);
    st.Stop();
    time_post = st.GetTime();

    LOGI("detect done, proc: %.3fms, inf: %.3fms, post: %.3fms || normalize: %.3fms, resize: %.3fms\n", time_proc, time_inf, time_post, time_norm, time_resize);
    return 1;
}
