#include <vector>
#include <opencv.hpp>
#include "TextureMapper.h"

/** 
** Assuming that source is a vector of cv::Mats with 3 channels for RGB.
**/
TextureMapper::TextureMapper(std::vector<cv::Mat> source) : source(source) {
    init();
    align(source, target);
    reconstruct();
}

void TextureMapper::align(std::vector<cv::Mat> source, std::vector<cv::Mat> target) {
    patchSearch();
    vote();
}

cv::Mat TextureMapper::patchSearch(int iterations, int patchSize) {

    // convert patch diameter to patch radius
    patchSize /= 2;

    // For each source pixel, output a 3-vector to the best match in
    // the target, with an error as the last channel.
    int sizes[3] = {source[0].size().width, source[0].size().height, /*number of frames*/ source.size()};
    cv::Mat out(3, sizes, CV_32FC(4));
    
    // Iterate over source frames, finding a match in the target where
    // the mask is high.
    
    for (int t = 0; t < source.size(); t++) {
        // INITIALIZATION - uniform random assignment
        for (int y = 0; y < source[0].size().height; y++) {
            for (int x = 0; x < source[0].size().width; x++) {
                int dx = randomInt(patchSize, target[0].size().width-patchSize-1);
                int dy = randomInt(patchSize, target[0].size().height-patchSize-1);
                int dt = randomInt(0, target.size()-1);
                unsigned char* p = out.ptr(x, y, t) + 0;
                *p = dx;
                p = out.ptr(x, y, t) + 1;
                *p = dy;
                p = out.ptr(x, y, t) + 2;
                *p = dt;
                p = out.ptr(x, y, t) + 3;
                *p = distance(x, y, t,
                                dx, dy, dt,
                                patchSize, HUGE_VAL);
            }
        }
    }
    
    bool forwardSearch = true;
    
    //Split the out matrix into 4 channels for dx, dy, dt, and error.
    std::vector<cv::Mat> channels(4);
    cv::split(out, channels);
    cv::Mat dx = channels[0], dy = channels[1], dt = channels[2], error = channels[3];
    
    for (int i = 0; i < iterations; i++) {
        //printf("Iteration %d\n", i);

        // PROPAGATION
        if (forwardSearch) {
            // Forward propagation - compare left, center and up
            for (int t = 0; t < source.size(); t++) {
                for (int y = 1; y < source[0].size().height; y++) {
                    for (int x = 1; x < source[0].size().width; x++) {
                        if (*error.ptr(x, y, t) + 0 > 0) {
                            float distLeft = distance(x, y, t,
                                                      (*dx.ptr(x-1, y, t)) + 1,
                                                      (*dy.ptr(x-1, y, t)),
                                                      (*dt.ptr(x-1, y, t)),
                                                      patchSize, *error.ptr(x, y, t));

                            if (distLeft < *error.ptr(x, y, t)) {
                                *dx.ptr(x, y, t) = (*dx.ptr(x-1, y, t))+1;
                                *dy.ptr(x, y, t) = *dy.ptr(x-1, y, t);
                                *dt.ptr(x, y, t) = *dt.ptr(x-1, y, t);
                                *error.ptr(x, y, t) = distLeft;
                            }

                            float distUp = distance(x, y, t,
                                                    *dx.ptr(x, y-1, t),
                                                    (*dy.ptr(x, y-1, t))+1,
                                                    *dt.ptr(x, y-1, t),
                                                    patchSize, *error.ptr(x, y, t));

                            if (distUp < *error.ptr(x, y, t)) {
                                *dx.ptr(x, y, t) = *dx.ptr(x, y-1, t);
                                *dy.ptr(x, y, t) = (*dy.ptr(x, y-1, t))+1;
                                *dt.ptr(x, y, t) = *dt.ptr(x, y-1, t);
                                *error.ptr(x, y, t) = distUp;
                            }
                        }

                        // TODO: Consider searching across time as well

                    }
                }
            }

        } else {
            // Backward propagation - compare right, center and down
            for (int t = source.size()-1; t >= 0; t--) {
                for (int y = source[0].size().height-2; y >= 0; y--) {
                    for (int x = source[0].size().width-2; x >= 0; x--) {
                        if (*error.ptr(x, y, t) > 0) {
                            float distRight = distance(x, y, t,
                                                       (*dx.ptr(x+1, y, t))-1,
                                                       *dy.ptr(x+1, y, t),
                                                       *dt.ptr(x+1, y, t),
                                                       patchSize, *error.ptr(x, y, t));

                            if (distRight < *error.ptr(x, y, t)) {
                                *dx.ptr(x, y, t) = (*dx.ptr(x+1, y, t))-1;
                                *dy.ptr(x, y, t) = (*dy.ptr(x+1, y, t);
                                *dt.ptr(x, y, t) = *dt.ptr(x+1, y, t);
                                *error.ptr(x, y, t) = distRight;
                            }

                            float distDown = distance(x, y, t,
                                                      *dx.ptr(x, y+1, t),
                                                      (*dy.ptr(x, y+1, t))-1,
                                                      *dt.ptr(x, y+1, t),
                                                      patchSize, *error.ptr(x, y, t));

                            if (distDown < *error.ptr(x, y, t)) {
                                *dx.ptr(x, y, t) = *dx.ptr(x, y+1, t);
                                *dy.ptr(x, y, t) = (*dy.ptr(x, y+1, t))-1;
                                *dt.ptr(x, y, t) = *dt.ptr(x, y+1, t);
                                *error.ptr(x, y, t) = distDown;
                            }
                        }

                        // TODO: Consider searching across time as well

                    }
                }
            }
        }

        forwardSearch = !forwardSearch;

        // RANDOM SEARCH
        for (int t = 0; t < source.size(); t++) {
            for (int y = 0; y < source[0].size().height; y++) {
                for (int x = 0; x < source[0].size().width; x++) {
                    if (*error.ptr(x, y, t) > 0) {

                        int radius = target[0].size().width > target[0].size().height ? target[0].size().width : target[0].size().height;

                        // search an exponentially smaller window each iteration
                        while (radius > 8) {
                            // Search around current offset vector (distance-weighted)

                            // clamp the search window to the image
                            int minX = (int)(*dx.ptr(x, y, t)) - radius;
                            int maxX = (int)(*dx.ptr(x, y, t)) + radius + 1;
                            int minY = (int)(*dy.ptr(x, y, t)) - radius;
                            int maxY = (int)(*dy.ptr(x, y, t)) + radius + 1;
                            if (minX < 0) { minX = 0; }
                            if (maxX > target[0].size().width) { maxX = target[0].size().width; }
                            if (minY < 0) { minY = 0; }
                            if (maxY > target[0].size().height) { maxY = target[0].size().height; }

                            int randX = randomInt(minX, maxX-1);
                            int randY = randomInt(minY, maxY-1);
                            int randT = randomInt(0, target.size() - 1);
                            float dist = distance(x, y, t,
                                                  randX, randY, randT,
                                                  patchSize, *error.ptr(x, y, t));
                            if (dist < *error.ptr(x, y, t)) {
                                *dx.ptr(x, y, t) = randX;
                                *dy.ptr(x, y, t) = randY;
                                *dt.ptr(x, y, t) = randT;
                                *error.ptr(x, y, t) = dist;
                            }

                            radius >>= 1;

                        }
                    }
                }
            }
        }
    }

    //Merge output channels back together
    std::vector<cv::Mat> outs = {dx, dy, dt, error};
    cv::merge(outs, out);

    return out;
    
}

float TextureMapper::distance(int sx, int sy, int st,
                                int tx, int ty, int tt,
                                int patchSize, int floatThreshold) {
                                    
    // Do not use patches on boundaries
    if (tx < patchSize || tx >= target[0].size().width-patchSize ||
        ty < patchSize || ty >= target[0].size().height-patchSize) {
        return HUGE_VAL;
    }
    
    // Compute distance between patches
    // Average L2 distance in RGB space
    float dist = 0;
    
    int x1 = max(-patchSize, -sx, -tx);
    int x2 = min(patchSize, -sx+source[0].size().width-1, -tx+target[0].size().width-1);
    int y1 = max(-patchSize, -sy, -ty);
    int y2 = min(patchSize, -sy+source[0].size().height-1, -ty+target[0].size().height-1);

    for (int c = 0; c < target[0].channels() /*color channels*/; c++) {
        for (int y = y1; y <= y2; y++) {
            for (int x = x1; x <= x2; x++) {
                
                uint8_t const* sourceValue_ptr(source[st].ptr(sx+x, sy+y) + c);
                uint8_t const* targetValue_ptr(target[tt].ptr(tx+x, ty+y) + c);
                
                float delta = *sourceValue_ptr - *targetValue_ptr;
                dist += delta * delta;

                // Early termination
                if (dist > threshold) {return HUGE_VAL;}
            }
        }
    }

    return dist;
}

void TextureMapper::vote() {

}

void TextureMapper::reconstruct() {

}

/**
** Initialize
** the targets and textures with their corresponding source images,
** i.e., Ti = Si and Mi = Si.
**/
void TextureMapper::init() {
    for (int t = 0; t < source.size(); t++) {
        target[t] = source[t].clone();
        texture[t] = source[t].clone();
    }
}