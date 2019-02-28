#ifndef TEXTUREMAPPER_H
#define TEXTUREMAPPER_H

class TextureMapper {

public:
    TextureMapper(std::vector<cv::Mat> source);

private:
    std::vector<cv::Mat> source;
    std::vector<cv::Mat> target;
    std::vector<cv::Mat> texture;

    void init();
    void align(std::vector<cv::Mat> source, std::vector<cv::Mat> target);
    void reconstruct();
    cv::Mat patchSearch(std::vector<cv::Mat> source, std::vector<cv::Mat> target, int iterations, int patchSize);
    void vote();
    float distance(std::vector<cv::Mat> source, std::vector<cv::Mat> target,
                                int sx, int sy, int st,
                                int tx, int ty, int tt,
                                int patchSize, int floatThreshold);
};

#endif