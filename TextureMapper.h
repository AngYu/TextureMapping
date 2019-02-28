#ifndef TEXTUREMAPPER_H
#define TEXTUREMAPPER_H
#include <vector>

class TextureMapper {

public:
    TextureMapper();
    void runMain(std::vector<cv::Mat> source);

private:    
    void init();
    void align();
    void reconstruct();
    void patchSearch();
    void vote();
    float distance();
};

#endif