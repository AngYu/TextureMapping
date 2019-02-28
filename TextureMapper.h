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
    cv::Mat patchSearch(int iterations, int patchSize);
    void vote();
    int Tixi();
    float distance(int sx, int sy, int st,
                    int tx, int ty, int tt,
                    int patchSize, float threshold);
    int randomInt(int min, int max);
};

#endif