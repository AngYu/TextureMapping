#include <iostream>
using namespace std;

int main() 
{
    //We initialize the targets and textures with their corresponding source images
    Ti = Si;
    Mi = Si;
    while (!converged) {
        align();
        reconstruct();
    }
    return 0;
}

void align(Image source, Image target) {
    //fix M1...Mn and minimize PBE by finding optimum T1...Tn
    //su and sv are the source patches overlapping with pixel xi
    //yu and yv refer to a single pixel in su and sv , respectively, corresponding to the xith pixel of the target image
    //Most of these variables are a function of the current pixel xi.
    const int patchSize = 7;

    patchSearch();
    voting();
}

int voting() {
    for(int image = 0; image < source.frames; image++) {
        for (int y = 0; y < source.height; y++) {
            for (int x = 0; x < source.width; x++) {
                int TiXi = TiXi(x, y, image);
            }
        }
    }
}

int TiXi(int x, int y, int t) {
    //image(x, y, t) is xi

    int usum = 0;
    for (int u = 0; u < completenessPatches; u++) {
        usum += sourcepatchu * yu;
    }
    int term1 = usum / 49;
    int vsum = 0;
    for (int v = 0; v < coherencePatches; v++) {
        vsum += sv * yv;
    }
    int term2 = (2 / 49) * vsum;
    int ksum = 0;
    for (int k = 0; k < N; k++) {
        ksum += Mk * (Xi->k);
    }
    int term3 = (0.1 / N) * wi * xi * ksum;
    int denominator = (U / 49) + ((2 * V) / 49) + (0.1 * wi * xi);
    return (term1 + term2 + term3) / denominator;
}

/**
 * Optimum texture is obtained by computing a weighted average
 * of all the projected targets
 **/
void reconstruct() {
    //fix T1...Tn and produce optimum texture at different views M1...Mn to minimize PBE.
    int jsum = 0;
    int jsumDenom = 0;
    for (int j = 0; j < N; j++) {
        jsum += wj * (Xi->j) * Tj * (Xi->j);
        jsumDenom += wj * (Xi->j);
    }
    int MiXi = jsum / jsumDenom;
}

Image patchSearch(Image source, Image target, Image mask, int iterations, int patchSize) {

    if (mask.defined()) {
        assert(target.width == mask.width &&
               target.height == mask.height &&
               target.frames == mask.frames,
               "Mask must have the same dimensions as the target\n");
        assert(mask.channels == 1,
               "Mask must have a single channel\n");
    }
    assert(iterations > 0, "Iterations must be a strictly positive integer\n");
    assert(patchSize >= 3 && (patchSize & 1), "Patch size must be at least 3 and odd\n");

    // convert patch diameter to patch radius
    patchSize /= 2;

    // For each source pixel, output a 3-vector to the best match in
    // the target, with an error as the last channel.
    Image out(source.width, source.height, source.frames, 4);

    // Iterate over source frames, finding a match in the target where
    // the mask is high

    for (int t = 0; t < source.frames; t++) {
        // INITIALIZATION - uniform random assignment
        for (int y = 0; y < source.height; y++) {
            for (int x = 0; x < source.width; x++) {
                int dx = randomInt(patchSize, target.width-patchSize-1);
                int dy = randomInt(patchSize, target.height-patchSize-1);
                int dt = randomInt(0, target.frames-1);
                out(x, y, t, 0) = dx;
                out(x, y, t, 1) = dy;
                out(x, y, t, 2) = dt;
                out(x, y, t, 3) = distance(source, target, mask,
                                           x, y, t,
                                           dx, dy, dt,
                                           patchSize, HUGE_VAL);
            }
        }
    }

    bool forwardSearch = true;

    Image dx = out.channel(0), dy = out.channel(1), dt = out.channel(2), error = out.channel(3);

    for (int i = 0; i < iterations; i++) {

        //printf("Iteration %d\n", i);

        // PROPAGATION
        if (forwardSearch) {
            // Forward propagation - compare left, center and up
            for (int t = 0; t < source.frames; t++) {
                for (int y = 1; y < source.height; y++) {
                    for (int x = 1; x < source.width; x++) {
                        if (error(x, y, t, 0) > 0) {
                            float distLeft = distance(source, target, mask,
                                                      x, y, t,
                                                      dx(x-1, y, t, 0)+1,
                                                      dy(x-1, y, t, 0),
                                                      dt(x-1, y, t, 0),
                                                      patchSize, error(x, y, t, 0));

                            if (distLeft < error(x, y, t, 0)) {
                                dx(x, y, t, 0) = dx(x-1, y, t, 0)+1;
                                dy(x, y, t, 0) = dy(x-1, y, t, 0);
                                dt(x, y, t, 0) = dt(x-1, y, t, 0);
                                error(x, y, t, 0) = distLeft;
                            }

                            float distUp = distance(source, target, mask,
                                                    x, y, t,
                                                    dx(x, y-1, t, 0),
                                                    dy(x, y-1, t, 0)+1,
                                                    dt(x, y-1, t, 0),
                                                    patchSize, error(x, y, t, 0));

                            if (distUp < error(x, y, t, 0)) {
                                dx(x, y, t, 0) = dx(x, y-1, t, 0);
                                dy(x, y, t, 0) = dy(x, y-1, t, 0)+1;
                                dt(x, y, t, 0) = dt(x, y-1, t, 0);
                                error(x, y, t, 0) = distUp;
                            }
                        }

                        // TODO: Consider searching across time as well

                    }
                }
            }

        } else {
            // Backward propagation - compare right, center and down
            for (int t = source.frames-1; t >= 0; t--) {
                for (int y = source.height-2; y >= 0; y--) {
                    for (int x = source.width-2; x >= 0; x--) {
                        if (error(x, y, t, 0) > 0) {
                            float distRight = distance(source, target, mask,
                                                       x, y, t,
                                                       dx(x+1, y, t, 0)-1,
                                                       dy(x+1, y, t, 0),
                                                       dt(x+1, y, t, 0),
                                                       patchSize, error(x, y, t, 0));

                            if (distRight < error(x, y, t, 0)) {
                                dx(x, y, t, 0) = dx(x+1, y, t, 0)-1;
                                dy(x, y, t, 0) = dy(x+1, y, t, 0);
                                dt(x, y, t, 0) = dt(x+1, y, t, 0);
                                error(x, y, t, 0) = distRight;
                            }

                            float distDown = distance(source, target, mask,
                                                      x, y, t,
                                                      dx(x, y+1, t, 0),
                                                      dy(x, y+1, t, 0)-1,
                                                      dt(x, y+1, t, 0),
                                                      patchSize, error(x, y, t, 0));

                            if (distDown < error(x, y, t, 0)) {
                                dx(x, y, t, 0) = dx(x, y+1, t, 0);
                                dy(x, y, t, 0) = dy(x, y+1, t, 0)-1;
                                dt(x, y, t, 0) = dt(x, y+1, t, 0);
                                error(x, y, t, 0) = distDown;
                            }
                        }

                        // TODO: Consider searching across time as well

                    }
                }
            }
        }

        forwardSearch = !forwardSearch;

        // RANDOM SEARCH
        for (int t = 0; t < source.frames; t++) {
            for (int y = 0; y < source.height; y++) {
                for (int x = 0; x < source.width; x++) {
                    if (error(x, y, t, 0) > 0) {

                        int radius = target.width > target.height ? target.width : target.height;

                        // search an exponentially smaller window each iteration
                        while (radius > 8) {
                            // Search around current offset vector (distance-weighted)

                            // clamp the search window to the image
                            int minX = (int)dx(x, y, t, 0) - radius;
                            int maxX = (int)dx(x, y, t, 0) + radius + 1;
                            int minY = (int)dy(x, y, t, 0) - radius;
                            int maxY = (int)dy(x, y, t, 0) + radius + 1;
                            if (minX < 0) { minX = 0; }
                            if (maxX > target.width) { maxX = target.width; }
                            if (minY < 0) { minY = 0; }
                            if (maxY > target.height) { maxY = target.height; }

                            int randX = randomInt(minX, maxX-1);
                            int randY = randomInt(minY, maxY-1);
                            int randT = randomInt(0, target.frames - 1);
                            float dist = distance(source, target, mask,
                                                  x, y, t,
                                                  randX, randY, randT,
                                                  patchSize, error(x, y, t, 0));
                            if (dist < error(x, y, t, 0)) {
                                dx(x, y, t, 0) = randX;
                                dy(x, y, t, 0) = randY;
                                dt(x, y, t, 0) = randT;
                                error(x, y, t, 0) = dist;
                            }

                            radius >>= 1;

                        }
                    }
                }
            }
        }
    }

    return out;
}