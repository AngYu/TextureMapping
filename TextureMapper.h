// A2DD.h
#ifndef A2DD_H
#define A2DD_H

class TextureMapper {
    public:
        TextureMapper();
        void runMain();
    private:
        init();
        align();
        reconstruct();
        patchSearch();
        voting();
        float distance();
}

#endif