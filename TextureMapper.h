#ifndef TEXTUREMAPPER_H
#define TEXTUREMAPPER_H

class TextureMapper {

public:
    TextureMapper(std::vector<cv::Mat> source, int patchSize = 7);

private:
    int patchSize;
    std::vector<cv::Mat> source;
    std::vector<cv::Mat> target;
    std::vector<cv::Mat> texture;

    void init();
    void align(std::vector<cv::Mat> source, std::vector<cv::Mat> target);
    void reconstruct();
    int Mixi();
    cv::Mat patchSearch(std::vector<cv::Mat> source, std::vector<cv::Mat> target, int iterations);
    void vote(cv::Mat completenessPatchMatches, cv::Mat coherencePatchMatches);
    std::vector<std::vector<std::vector<int>>> findSourcePatches(cv::Mat completenessPatchMatches, cv::Mat coherencePatchMatches, 
                                                                    int x, int y, int t);
    bool isInTargetPatch(cv::Vec<float, 4> targetMatch, int x, int y, int t);
    int Tixi(std::vector<std::vector<int>> completenessPatches, std::vector<std::vector<int>> coherencePatches, int c);
    float distance(int sx, int sy, int st,
                    int tx, int ty, int tt,
                    int patchSize, float threshold);
    int randomInt(int min, int max);
};

#endif