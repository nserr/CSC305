/* 
 * CSC 305 Assignment 1
 * Noah Serr
 * V00891494
 */

#include "assignment.hpp"
#include <random>

 /**
  * Multisample antialiasing using Regular Sampling.
  *
  * @param x The x value of the pixel
  * @param y The y value of the pixel
  * @param planes The array of planes
  * @param num_planes The number of planes in the array
  * @param spheres The array of spheres
  * @param num_spheres The number of spheres in the array
  * @param triangles The array of triangles
  * @param num_triangles The number of triangles in the array
  * return the average of all sample colours
  */
Colour regularSample(std::size_t x, std::size_t y, Plane planes[], int num_planes, Sphere spheres[], int num_spheres, Triangle triangles[], int num_triangles) {
    atlas::math::Ray<atlas::math::Vector> ray{ {0,0,0}, {0,0,-1} };
    Colour colour_sum_reg{ 0,0,0 };
    ShadeRec trace_data{};
    bool one_hit = false;

    // Take centre of pixel samples as a 4x4 grid
    for (float sample_x{ 0.125 }; sample_x < 1.0f; sample_x += 0.25f) {
        for (float sample_y{ 0.125 }; sample_y < 1.0f; sample_y += 0.25f) {
            ray.o = { x + sample_x, y + sample_y, 0 };

            // Check each sphere
            for (int i{ 0 }; i < num_spheres; i++) {
                // If ray hits, track colour of pixel sample
                if (spheres[i].hit(ray, trace_data)) {
                    colour_sum_reg += trace_data.colour;
                    one_hit = true;
                }      
            }

            // Check each triangle
            for (int i{ 0 }; i < num_triangles; i++) {
                // If ray hits, track colour of pixel sample
                if (triangles[i].hit(ray, trace_data)) {
                    colour_sum_reg += trace_data.colour;
                    one_hit = true;
                }
            }

            // If no other objects intersect, check planes
            if (one_hit == false) {
                for (int i{ 0 }; i < num_planes; i++) {
                    // If ray hits, track colour of pixel sample
                    if (planes[i].hit(ray, trace_data)) {
                        colour_sum_reg += trace_data.colour;
                    }
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
 * @param planes The array of planes
 * @param num_planes The number of planes in the array
 * @param spheres The array of spheres
 * @param num_spheres The number of spheres in the array
 * @param triangles The array of triangles
 * @param num_triangles The number of triangles in the array
 * @param num_points The number of points to sample
 * return the average of all sample colours
 */
Colour randomSample(std::size_t x, std::size_t y, Plane planes[], int num_planes, Sphere spheres[], int num_spheres, Triangle triangles[], int num_triangles, int num_points) {
    atlas::math::Ray<atlas::math::Vector> ray{ {0,0,0}, {0,0,-1} };
    Colour colour_sum_ran{ 0,0,0 };
    ShadeRec trace_data{};
    bool one_hit = false;

    // Collect random points within the pixel
    for (int i{ 0 }; i < num_points; i++) {
        float sample_x = ((float)rand() / RAND_MAX);
        float sample_y = ((float)rand() / RAND_MAX);

        ray.o = { x + sample_x, y + sample_y, 0 };

        // Check each object
        for (int j{ 0 }; j < num_spheres; j++) {
            // If ray hits, track colour of pixel sample
            if (spheres[j].hit(ray, trace_data)) {
                colour_sum_ran += trace_data.colour;
                one_hit = true;
            }
        }

        // Check each triangle
        for (int k{ 0 }; k < num_triangles; k++) {
            // If ray hits, track colour of pixel sample
            if (triangles[k].hit(ray, trace_data)) {
                colour_sum_ran += trace_data.colour;
                one_hit = true;
            }
        }

        // If no other objects intersect, check planes
        if (one_hit == false) {
            for (int l{ 0 }; l < num_planes; l++) {
                // If ray hits, track colour of pixel sample
                if (planes[l].hit(ray, trace_data)) {
                    colour_sum_ran += trace_data.colour;
                }
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

    constexpr Plane p1{ {0,400,100}, {0,1,1}, {0.67,0,1} };
    constexpr Plane p2{ {200,0,100}, {1,0,0}, {1,0,1} };
    constexpr Sphere s1{ {100,75,0}, 50, {1,0,0} };
    constexpr Sphere s2{ {280,150,0}, 75, {0,1,0} };
    constexpr Sphere s3{ {320,450,0}, 75, {1,0.5,0} };
    constexpr Sphere s4{ {500,525,0}, 50, {1,0,0.5} };
    constexpr Triangle t1{ {475, 175, 0}, {400, 325, 0}, {550, 325, 0}, {0,0,1} };
    constexpr Triangle t2{ {100, 300, 0}, {50, 450, 0}, {150, 450, 0}, {0,1,1} };
    
    constexpr int num_planes = 2;
    constexpr int num_spheres = 4;
    constexpr int num_triangles = 2;

    Plane planes[num_planes] = { p1, p2 };
    Sphere spheres[num_spheres] = { s1, s2, s3, s4 };
    Triangle triangles[num_triangles] = { t1, t2 };
   
    std::vector<Colour> image{ image_width * image_height };

    // Loop through entire image
    for (std::size_t y{ 0 }; y < image_height; y++) {
        for (std::size_t x{ 0 }; x < image_width; x++) {

            image[x + y * image_height] = regularSample(x, y, planes, num_planes, spheres, num_spheres, triangles, num_triangles);
            //image[x + y * image_height] = randomSample(x, y, planes, num_planes, spheres, num_spheres, triangles, num_triangles, 16);
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
