/* 
 * CSC 305 Assignment 1
 * Noah Serr
 * V00891494
 */

#include "assignment.hpp"
#include <random>

 /**
  * Multisample antialiasing using Random Sampling.
  *
  * @param x The x value of the pixel
  * @param y The y value of the pixel
  * @param arr The array of objects
  * @param arrSize The number of objects in the array
  * return the average of all sample colours
  */
Colour regularSample(std::size_t x, std::size_t y, Sphere arr[], int arrSize) {
    atlas::math::Ray<atlas::math::Vector> ray{ {0,0,0}, {0,0,-1} };
    Colour colour_sum_reg{ 0,0,0 };
    ShadeRec trace_data{};

    // Take centre of pixel samples as a 4x4 grid
    for (float sample_x{ 0.125 }; sample_x < 1.0f; sample_x += 0.25f) {
        for (float sample_y{ 0.125 }; sample_y < 1.0f; sample_y += 0.25f) {
            ray.o = { x + sample_x, y + sample_y, 0 };

            // Check each object
            for (int i{ 0 }; i < arrSize; i++) {
                // If ray hits, track colour of pixel sample
                if (arr[i].hit(ray, trace_data)) {
                    colour_sum_reg += trace_data.colour;
                }
            }
        }
    }

    // Take average of all sample colours
    colour_sum_reg /= 16;
    
    return colour_sum_reg;
}

/**
 * Multisample antialiasing using Random Sampling.
 *
 * @param x The x value of the pixel
 * @param y The y value of the pixel
 * @param arr The array of objects
 * @param arrSize The number of objects in the array
 * @param num_points The number of random points to use
 * return the average of all sample colours
 */
Colour randomSample(std::size_t x, std::size_t y, Sphere arr[], int arrSize, int num_points) {
    atlas::math::Ray<atlas::math::Vector> ray{ {0,0,0}, {0,0,-1} };
    Colour colour_sum_ran{ 0,0,0 };
    ShadeRec trace_data{};

    // Collect random points within the pixel
    for (int i{ 0 }; i < num_points; i++) {
        float sample_x = ((float)rand() / RAND_MAX);
        float sample_y = ((float)rand() / RAND_MAX);

        ray.o = { x + sample_x, y + sample_y, 0 };

        // Check each object
        for (int j{ 0 }; j < arrSize; j++) {
            // If ray hits, track colour of pixel sample
            if (arr[j].hit(ray, trace_data)) {
                colour_sum_ran += trace_data.colour;
            }
        }
    }

    // Take average of all sample colours
    colour_sum_ran /= num_points;

    return colour_sum_ran;
}

int main()
{
    constexpr std::size_t image_width{ 600 };
    constexpr std::size_t image_height{ 600 };
    constexpr Colour background{ 0,0,0 };

    constexpr Sphere s1{ {100,75,0}, 50, {1,0,0} };
    constexpr Sphere s2{ {280,150,0}, 75, {0,1,0} };
    constexpr Sphere s3{ {480,225,0}, 100, {0,0,1} };
    constexpr Sphere s4{ {120,375,0}, 100, {0.5,0,1} };
    constexpr Sphere s5{ {320,450,0}, 75, {0,1,1} };
    constexpr Sphere s6{ {500,525,0}, 50, {1,0,0.5} };
    Sphere arr[6] = {s1, s2, s3, s4, s5, s6};
   
    std::vector<Colour> image{ image_width * image_height };

    // Loop through entire image
    for (std::size_t y{ 0 }; y < image_height; y++) {
        for (std::size_t x{ 0 }; x < image_width; x++) {

            image[x + y * image_height] = regularSample(x, y, arr, 6);
            //image[x + y * image_height] = randomSample(x, y, arr, 6, 16);

            // Have randomSample commented out since regularSample appears to give a better image.
            // Both have been implemented and work nicely.
        }
    }

    saveToBMP("C:/Users/noahs/OneDrive/Desktop/School/CSC 305/Assignments/A1/bundle/output.bmp", image_width, image_height, image);
}

/**
 * Saves a BMP image file based on the given array of pixels. All pixel values
 * have to be in the range [0, 1].
 *
 * @param filename The name of the file to save to.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param image The array of pixels representing the image.
 */
void saveToBMP(std::string const& filename,
               std::size_t width,
               std::size_t height,
               std::vector<Colour> const& image)
{
    std::vector<unsigned char> data(image.size() * 3);

    for (std::size_t i{0}, k{0}; i < image.size(); ++i, k += 3)
    {
        Colour pixel = image[i];
        data[k + 0]  = static_cast<unsigned char>(pixel.r * 255);
        data[k + 1]  = static_cast<unsigned char>(pixel.g * 255);
        data[k + 2]  = static_cast<unsigned char>(pixel.b * 255);
    }

    stbi_write_bmp(filename.c_str(),
                   static_cast<int>(width),
                   static_cast<int>(height),
                   3,
                   data.data());
}
