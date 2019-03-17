#include <vector>
#include <opencv.hpp>
#include "TextureMapper.h"

/** 
** Assuming that source is a vector of cv::Mats
**/
TextureMapper::TextureMapper(PlyModel model, std::vector<cv::Mat> source, std::vector<cv::Mat> TcwPoses, int patchSize = 7) : source(source), TcwPoses(TcwPoses), patchSize(patchSize) {
    init();
    align(source, target);
    reconstruct();
}

void TextureMapper::align(std::vector<cv::Mat> source, std::vector<cv::Mat> target) {
    int iterations = 1;
    cv::Mat completenessPatchMatches = patchSearch(source, target, iterations);
    cv::Mat coherencePatchMatches = patchSearch(target, source, iterations);
    vote(completenessPatchMatches, coherencePatchMatches);
}

cv::Mat TextureMapper::patchSearch(std::vector<cv::Mat> source, std::vector<cv::Mat> target, int iterations) {

    // convert patch diameter to patch radius
    patchSize /= 2;

    // For each source pixel, output a 3-vector to the best match in
    // the target, with an error as the last channel. The 3-vector should be the location of the patch center.
    int sizes[3] = {source[0].size().width, source[0].size().height, /*number of frames*/ source.size()};
    cv::Mat out(3, sizes, CV_32FC(4));
    
    // Iterate over source frames, finding a match in the target where
    // the mask is high.
    
    for (int t = 0; t < source.size(); t++) {
        // INITIALIZATION - uniform random assignment of out matrix values.
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

void TextureMapper::vote(cv::Mat completenessPatchMatches, cv::Mat coherencePatchMatches) {
    //For each pixel in the target
    for (int t = 0; t < target.size(); t++) {
        for (int y = 0; y < target[0].size().height; y++) {
            for (int x = 0; x < target[0].size().width; x++) {
                std::vector<std::vector<std::vector<int>>> patches = findSourcePatches(completenessPatchMatches, coherencePatchMatches, x, y, t);
                std::vector<std::vector<int>> completenessPatches = patches[0];
                std::vector<std::vector<int>> coherencePatches = patches[1];
                
                for (int c = 0; c < source[0].channels(); c++) {
                    Tixi(completenessPatches, coherencePatches, c);
                }

            }
        }
    }
}

std::vector<std::vector<std::vector<int>>> TextureMapper::findSourcePatches(cv::Mat completenessPatchMatches, cv::Mat coherencePatchMatches, int x, int y, int t) {
    std::vector<std::vector<std::vector<int>>> sourcePatches;
    std::vector<std::vector<int>> completenessPatches;
    sourcePatches[0] = completenessPatches;
    std::vector<std::vector<int>> coherencePatches;
    sourcePatches[1] = coherencePatches;
    //Find patches in target that contain the pixel
    int targetx = x;
    int targety = y;
    int x1 = std::max(-patchSize, -targetx);
    int x2 = std::min(patchSize, -targetx+target[0].size().width-1);
    int y1 = std::max(-patchSize, -targety);
    int y2 = std::min(patchSize, -targety+target[0].size().height-1);
    
    //Completeness: Find Source patches that have target patches as their most similar patch
    //For each pixel in completenessPatchMatches
    for (int st = 0; st < source.size(); st++) {
        for (int sy = 0; sy < source[0].size().height; sy++) {
            for (int sx = 0; sx < source[0].size().width; sx++) {
                cv::Vec<float, 4> patchMatch = completenessPatchMatches.at<cv::Vec<float, 4>>(st, sy, st);
                int stx = patchMatch[0], sty = patchMatch[1], stt = patchMatch[2];
                if ( /* is in x range */(stx >= x1 && stx <= x2) && /** is in y range */ (sty >= y1 && sty <= y2) && stt == t) {
                    //return value of the target pixel within the source patch
                    std::vector<int> targetPixel;
                    //Find target pixel in source patch
                    int targetPixelX = (x - stx) + sx;
                    int targetPixelY = (y - sty) + sy;
                    for (int c = 0; c < source[0].channels(); c++) {
                        targetPixel.push_back(source[st].at<cv::Vec<float, 4>>(targetPixelX, targetPixelY)[c]);
                    }
                    sourcePatches[0].push_back(targetPixel);
                }
            }
        }
    }

    //Coherence: Find the Source patches most similar to the target patches
    for (int patchy = y1; patchy <= y2; patchy++) {
        for (int patchx = x1; patchx <= x2; patchx++) {
            cv::Vec<float, 4> sourcePatchVec = coherencePatchMatches.at<cv::Vec<float, 4>>(patchx, patchy, t);
            //return value of the target pixel within the source patch
            std::vector<int> targetPixel;
            //Find target pixel in source patch
            int targetPixelX = (x - patchx) + sourcePatchVec[0];
            int targetPixelY = (y - patchy) + sourcePatchVec[1];
            for (int c = 0; c < source[0].channels(); c++) {
                targetPixel.push_back(source[sourcePatchVec[2]].at<cv::Vec<float, 4>>(targetPixelX, targetPixelY)[c]);
            }
            sourcePatches[0].push_back(targetPixel);
        }
    }

    return sourcePatches;
}

int TextureMapper::Tixi(std::vector<std::vector<int>> completenessPatches, std::vector<std::vector<int>> coherencePatches, int c /*color channel*/) {
    //su and sv are the source patches overlapping with pixel xi of the target for the completeness and coherence terms, respectively.
    //yu and yv refer to a single pixel in su and sv , respectively, corresponding to the Xith pixel of the target image. 
    //U and V refer to the number of patches for the completeness and coherence terms, respectively.
    //wj = (cos(θ)**2) / (d**2), where θ is the angle between the surface
    //normal and the viewing direction at image j and d denotes the distance between the camera and the surface.
    int U = completenessPatches.size();
    int V = coherencePatches.size();
    int L = 49; //L is the number of pixels in a patch (7 x 7 = 49)
    int alpha = 2;
    int lambda = 0.1;
    int sum1 = 0;
    int N = texture.size(); //N is the number of texture images.
    for (int u = 0; u < U; u++) {
        int upatch = completenessPatches[u][c];
        sum1 += upatch;
    }
    int term1 = (1/L)*sum1;
    int sum2 = 0;
    for (int v; v < V; v++) {
        int vpatch = coherencePatches[v][c];
        sum2 += vpatch;
    }
    int term2 = (alpha / L) * sum2;
    int sum3 = 0;
    for (int k = 0; k < N; k++) {
        //Mk(Xi->k) RGB color of the kth texture at pixel Xi->k, i.e., the result of projecting texture k to camera i
        // (Xi->k is pixel position projected from image i to k)
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
    int N = texture.size();
    int numerator = 0;
    for (int j = 0; j < N; j++) {
        //Tj(Xi->j) is the result of projecting target j to camera i
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

std::vector<cv::Mat> TextureMapper::getRGBD(std::vector<cv::Mat> target, std::vector<cv::Mat> TcwPoses, PlyModel model) {
    //Get depth for all of the pixels. This will either require rasterization or ray-tracing (I need to do more research to determine which one).
    
}

bool TextureMapper::projectToSurface(MeshDocument &md, RichParameterSet & par, vcg::CallBackPos *cb) {
    // accumulation buffers for colors and weights
    int buff_ind;
    double *weights;
    double *acc_red;
    double *acc_grn;
    double *acc_blu;

    // init accumulation buffers for colors and weights
    acc_red = new double[model->cm.vn];
    acc_grn = new double[model->cm.vn];
    acc_blu = new double[model->cm.vn];
    for(int buff_ind=0; buff_ind<model->cm.vn; buff_ind++)
    {
        acc_red[buff_ind] = 0.0;
        acc_grn[buff_ind] = 0.0;
        acc_blu[buff_ind] = 0.0;
    }

    //for each camera
    for (int cam = 0; cam < TcwPoses.size(); cam++) {
        //if raster is good
            glContext->makeCurrent();
            
            // render normal & depth
            rendermanager->renderScene(raster->shot, model, RenderHelper::NORMAL, glContext, my_near[cam_ind]*0.5, my_far[cam_ind]*1.25);

            // Unmaking context current
            glContext->doneCurrent();


            //THIS IS WHERE THE SEARCH FOR VERTICES IS
            // For vertex in model
            for (int vertex; vertex < ; vertex++) {
                //project point to image space
                //get vector from the point-to-be-colored to the camera center
                //if inside image
                    // add color buffers
            } //end for each vertex
    } //end for each camera
    // Paint model vertices with colors
}

int max(int x, int y, int z) {
    return std::max(std::max(x, y), z);
}

int min(int x, int y, int z){
    return std::min(std::min(x, y), z);
}