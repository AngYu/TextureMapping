#ifndef TEXTUREMAPPER_H
#define TEXTUREMAPPER_H
#include "PatchMatch.h"

class TextureMapper {

public:
    TextureMapper();
    void runMain();

private:    
    void init();
    void align();
    void reconstruct();
    void patchSearch();
    void vote();
    float distance();
};

#endif