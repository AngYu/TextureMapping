#include <vector>
#include <opencv.hpp>
#include "TextureMapper.h"

/** 
** Assuming that source is a vector of cv::Mats
**/
TextureMapper::TextureMapper(std::vector<cv::Mat> source) : source(source) {
    init();
    align(source, target);
    reconstruct();
}

void TextureMapper::align(std::vector<cv::Mat> source, std::vector<cv::Mat> target) {
    int iterations = 1;
    int patchSize = 7;
    cv::Mat out = patchSearch(iterations, patchSize);
    vote(out);
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
                                *dy.ptr(x, y, t) = *dy.ptr(x+1, y, t);
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
                                int patchSize, float threshold) {
                                    
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

void TextureMapper::vote(cv::Mat patchSearchResult) {
    for (int t = 0; t < target.size(); t++) {
        for (int y = 0; y < target[0].size().height; y++) {
            for (int x = 0; x < target[0].size().width; x++) {
                //Get the source patches overlapping with pixel (x, y, t) of the target.
                std::vector<cv::Mat> patches;
                *target[t].ptr(x,y) = Tixi(patches);
            }
        }
    }
}

int TextureMapper::Tixi(std::vector<cv::Mat> patches) {
    //L is the number of pixels in a patch (7 x 7 = 49)
    //su and sv are the source patches overlapping with pixel xi of the target for the completeness and coherence terms, respectively.
    //yu and yv refer to a single pixel in su and sv , respectively, corresponding to the Xith pixel of the target image. 
    //U and V refer to the number of patches for the completeness and coherence terms, respectively.
    //wj = (cos(θ)**2) / (d**2), where θ is the angle between the surface
    //normal and the viewing direction at image j and d denotes the distance between the camera and the surface.
    //I believe N is the number of images.
    int U = patches.size();
    int V = patches.size();
    int L = 49;
    int alpha = 2;
    int lambda = 0.1;
    int sum1 = 0;
    for (int u = 0; u < U; u++) {
        sum1 += su(yu);
    }
    int term1 = (1/L)*sum1;
    int sum2 = 0;
    for (int v; v < V; v++) {
        sum2 += sv(yv);
    }
    int term2 = (alpha / L) * sum2;
    int sum3 = 0;
    for (int k = 0; k < N; k++) {
        sum3 += Mk(Xi->k);
    }
    int term3 = (lambda / N) * wi(xi) * sum3;
    int denominator = (U / L) + ((alpha * V) / L) + (lambda * wi(xi));
    return ((term1 + term2 + term3) / denominator);
}

void TextureMapper::reconstruct() {
    for (int t = 0; t < texture.size(); t++) {
        for (int y = 0; y < texture[0].size().height; y++) {
            for (int x = 0; x < texture[0].size().width; x++) {
                *texture[t].ptr(x,y) = Mixi();
            }
        }
    }
}

int TextureMapper::Mixi() {
    int numerator = 0;
    for (int j = 0; j < N; j++) {
        numerator += wj(Xi->j) * Tj(Xi->j);
    }
    int denominator = 0;
    for (int j = 0; j < N; j++) {
        denominator += wj(Xi->j);
    }
    return numerator / denominator;
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

int TextureMapper::randomInt(int min, int max) {
    return min + (rand() % static_cast<int>(max - min + 1));
}

int max(int x, int y, int z) {
    return std::max(std::max(x, y), z);
}

int min(int x, int y, int z){
    return std::min(std::min(x, y), z);
}