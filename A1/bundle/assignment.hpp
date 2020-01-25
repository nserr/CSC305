/*
 * CSC 305 Assignment 1
 * Noah Serr
 * V00891494
 */

#pragma once

#include <atlas/math/Math.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <vector>

using namespace atlas;
using Colour = math::Vector;


void saveToBMP(std::string const& filename,
               std::size_t width,
               std::size_t height,
               std::vector<Colour> const& image);


struct ShadeRec {
    Colour colour;
    float t;
};

class Sphere {
public:
    constexpr Sphere(atlas::math::Point center, float radius, Colour colour):
        center_{center},
        radius_{radius},
        radius_sqr_{radius * radius},
        colour_{colour}
    {}

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& trace_data) const {
        auto o_c{ ray.o - center_ };

        auto a{ glm::dot(ray.d, ray.d) };
        auto b{ glm::dot(ray.d, o_c) * 2 };
        auto c{ glm::dot(o_c, o_c) - radius_sqr_ };

        auto roots{ b * b - (4.0f * a * c) };

        if (roots >= 0.0f) {
            trace_data.t = ((-b - std::sqrt(roots)) / (2.0f * a));
            auto norm{ center_ - (ray.d * trace_data.t + ray.o) };
            trace_data.colour = colour_;
            return true;
        }

        return false;
    }
        
private:
    atlas::math::Point center_;
    float radius_, radius_sqr_;
    Colour colour_;
};