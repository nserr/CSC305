/*
 * CSC 305 Assignment 1
 * Noah Serr
 * V00891494
 */

#pragma once

#include <atlas/math/Math.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/math/Solvers.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <vector>
#include <iostream>
#include <math.h>
using namespace std;

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

class Plane {
public:
    constexpr Plane(atlas::math::Point p, atlas::math::Normal norm, Colour colour):
        p_{p},
        norm_{norm},
        colour_{colour}
    {}

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& trace_data) const {
        auto o_c{ p_ - ray.o };
        auto a{ glm::dot(o_c, norm_) };
        auto b = glm::dot(ray.d, norm_);
        auto t{ a / b };

        if (t > 0.0001f) {
            trace_data.colour = colour_;
            trace_data.t = t;
            return true;
        }
        
        return false;
    }

private:
    atlas::math::Point p_;
    atlas::math::Vector norm_;
    Colour colour_;
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

class Triangle {
public:
    constexpr Triangle(atlas::math::Point p1, atlas::math::Point p2, atlas::math::Point p3, Colour colour) :
        a_{p1},
        b_{p2},
        c_{p3},
        colour_{colour}
    {}

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& trace_data) const {
        atlas::math::Vector norm_num{ glm::cross((b_ - a_), (c_ - a_)) };
        atlas::math::Vector norm_denom{ glm::length(glm::cross((b_ - a_), (c_ - a_))) };
        atlas::math::Vector norm = norm_num / norm_denom;

        float dot = glm::dot(norm, ray.d);
        if (fabs(dot) < 0.0001f) {
            return false;
        }

        float d = glm::dot(norm, a_);
        float t = -(glm::dot(norm, ray.o) + d) / dot;

        atlas::math::Vector cross;
        atlas::math::Vector p{ (ray.o.x + t + ray.d.x), (ray.o.y + t + ray.d.y), (ray.o.z + t + ray.d.z) };

        atlas::math::Vector e1{ (b_.x - a_.x), (b_.y - a_.y), (b_.z - a_.z) };
        atlas::math::Vector vp1{ (p.x - a_.x), (p.y - a_.y), (p.z - a_.z) };

        cross = glm::cross(e1, vp1);
        if (glm::dot(norm, cross) < 0) {
            return false;
        }

        atlas::math::Vector e2{ (c_.x - b_.x), (c_.y - b_.y), (c_.z - b_.z) };
        atlas::math::Vector vp2{ (p.x - b_.x), (p.y - b_.y), (p.z - b_.z) };

        cross = glm::cross(e2, vp2);
        if (glm::dot(norm, cross) < 0) {
            return false;
        }

        atlas::math::Vector e3{ (a_.x - c_.x), (a_.y - c_.y), (a_.z - c_.z) };
        atlas::math::Vector vp3{ (p.x - c_.x), (p.y - c_.y), (p.z - c_.z) };

        cross = glm::cross(e3, vp3);
        if (glm::dot(norm, cross) < 0) {
            return false;
        }

        trace_data.colour = colour_;
        trace_data.t = t;
        return true;
    }

private:
    atlas::math::Point a_;
    atlas::math::Point b_;
    atlas::math::Point c_;
    Colour colour_;
};