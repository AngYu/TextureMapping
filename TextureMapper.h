#ifndef TEXTUREMAPPER_H
#define TEXTUREMAPPER_H

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