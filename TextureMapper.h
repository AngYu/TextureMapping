// A2DD.h
#ifndef A2DD_H
#define A2DD_H

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
}

#endif