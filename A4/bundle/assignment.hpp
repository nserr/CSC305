/*
 * CSC 305 Assignment 4
 * Noah Serr
 * V00891494
 */

#pragma once

#include <atlas/math/Math.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/core/Float.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <vector>
#include <limits>
#include <memory>

using atlas::core::areEqual;

using Colour = atlas::math::Vector;


void saveToBMP(std::string const& filename,
    std::size_t width,
    std::size_t height,
    std::vector<Colour> const& image);

class BRDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;

struct World {
    std::size_t width, height;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
    std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
    int maxDepth;
};

struct ShadeRec {
    Colour colour;
    float t;
    atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
    int depth;
};

class Camera {
public:
    Camera();
    virtual ~Camera() = default;

    virtual void renderScene(std::shared_ptr<World> world) const = 0;

    void setEye(atlas::math::Point const& eye);
    void setLookAt(atlas::math::Point const& lookAt);
    void setUpVector(atlas::math::Point const& up);

    void computeUVW();

protected:
    atlas::math::Point mEye;
    atlas::math::Point mLookAt;
    atlas::math::Point mUp;
    atlas::math::Vector mU, mV, mW;

};

class Sampler {
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndices();

    virtual void generateSamples() = 0;

    atlas::math::Point sampleUnitSquare();

    void mapSamplesToHemisphere(const float e);

protected:
    std::vector<atlas::math::Point> mSamples;
    std::vector<int> mShuffledIndices;
    std::vector<atlas::math::Point> hemisphereSamples;

    int mNumSamples;
    int mNumSets;
    unsigned long mCount;
    int mJump;
};

class Shape {
public:
    Shape();
    virtual ~Shape() = default;

    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

    void setColour(Colour const& col);

    Colour getColour() const;

    void setMaterial(std::shared_ptr<Material> const& material);

    std::shared_ptr<Material> getMaterial() const;

    virtual bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const = 0;

protected:
    virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const = 0;

    Colour mColour;
    std::shared_ptr<Material> mMaterial;
};

class BRDF {
public:
    virtual ~BRDF() = default;

    virtual Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const = 0;

    virtual Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const = 0;
};

class Material {
public:
    virtual ~Material() = default;

    virtual Colour shade(ShadeRec& sr) = 0;
};

class Light {
public:
    virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

    virtual Colour L(ShadeRec& sr);

    void scaleRadiance(float b);

    void setColour(Colour const& c);

    virtual bool castsShadows() = 0;

    virtual bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const = 0;

    virtual void setSampler(std::shared_ptr<Sampler> sPtr) = 0;

protected:
    Colour mColour;
    float mRadiance;
};

class Plane : public Shape {
public:
    Plane(atlas::math::Point p, atlas::math::Normal norm);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

    bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

    atlas::math::Point p_;
    atlas::math::Normal norm_;
};

class Sphere : public Shape {
public:
    Sphere(atlas::math::Point center, float radius);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

    bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

    atlas::math::Point mCenter;
    float mRadius;
    float mRadiusSqr;
};

class Triangle : public Shape {
public:
    Triangle(atlas::math::Point p1, atlas::math::Point p2, atlas::math::Point p3);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const;

    bool shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const;

    atlas::math::Point a_;
    atlas::math::Point b_;
    atlas::math::Point c_;
};

class Pinhole : public Camera {
public:
    Pinhole();

    void setDistance(float distance);
    void setZoom(float zoom);

    atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
    void renderScene(std::shared_ptr<World> world) const;

private:
    float mDistance;
    float mZoom;
};

class Regular : public Sampler {
public:
    Regular(int numSamples, int numSets);

    void generateSamples();
};

class Random : public Sampler {
public:
    Random(int numSamples, int numSets);

    void generateSamples();
};

class Jittered : public Sampler {
public:
    Jittered(int numSamples, int numSets);

    void generateSamples();
};

class Lambertian : public BRDF {
public:
    Lambertian();
    Lambertian(Colour diffuseColour, float diffuseReflection);

    Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const override;

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

    void setDiffuseReflection(float kd);

    void setDiffuseColour(Colour const& colour);

    Colour sampleF(ShadeRec const& sr, atlas::math::Vector& wo, atlas::math::Vector& wi) const;

private:
    Colour mDiffuseColour;
    float mDiffuseReflection;
};

class GlossySpecular : public BRDF {
public:
    GlossySpecular();
    GlossySpecular(float ks, Colour cs, float exp);

    Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const;

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const;

    void setSpecularReflection(float ks);

    void setSpecularColour(Colour cs);

    void setExp(float exp);

private:
    float mKs;
    Colour mCs;
    float mExp;
};

class Matte : public Material {
public:
    Matte();
    Matte(float kd, float ka, Colour colour);

    void setDiffuseReflection(float k);

    void setAmbientReflection(float k);

    void setDiffuseColour(Colour colour);

    Colour shade(ShadeRec& sr);

private:
    std::shared_ptr<Lambertian> mDiffuseBRDF;
    std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Phong : public Material {
public:
    Phong();
    Phong(float ka, float kd, Colour c, float ks, Colour cs, float exp);

    void setDiffuseReflection(float dr);

    void setAmbientReflection(float ar);

    void setDiffuseColour(Colour c);

    void setSpecularReflection(float ks);

    void setSpecularColour(Colour cs);

    void setExp(float exp);

    Colour shade(ShadeRec& sr);

private:
    Lambertian ambient;
    Lambertian diffuse;
    GlossySpecular specular;
};

class Directional : public Light {
public:
    Directional();
    Directional(atlas::math::Vector const& d);

    void setDirection(atlas::math::Vector const& d);

    atlas::math::Vector getDirection(ShadeRec& sr) override;
    
    bool castsShadows() override;

    bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const override;

    void setSampler(std::shared_ptr<Sampler> sPtr) override;

private:
    atlas::math::Vector mDirection;
};

class Point : public Light {
public:
    Point();
    Point(atlas::math::Vector const& loc);

    void setLocation(atlas::math::Point const& loc);

    atlas::math::Point getLocation();

    atlas::math::Vector getDirection(ShadeRec& sr) override;

    bool castsShadows() override;

    bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const override;

    void setSampler(std::shared_ptr<Sampler> sPtr) override;

private:
    atlas::math::Vector mLoc;
};

class Ambient : public Light {
public:
    Ambient();

    atlas::math::Vector getDirection(ShadeRec& sr) override;

    bool castsShadows() override;

    bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const override;

    void setSampler(std::shared_ptr<Sampler> sPtr) override;

private:
    atlas::math::Vector mDirection;
};

class AmbientOccluder : public Light {
public:
    AmbientOccluder();

    void setSampler(std::shared_ptr<Sampler> sPtr) override;

    atlas::math::Vector getDirection(ShadeRec& sr) override;

    bool inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const override;

    Colour L(ShadeRec& sr) override;

    bool castsShadows() override;

private:
    atlas::math::Vector u, v, w;
    std::shared_ptr<Sampler> mSPtr;
    Colour minAmount{ 0.0f, 0.0f, 0.0f };
};

class Whitted {
    Whitted(ShadeRec& sr);

    Colour trace_ray(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const;

private:
    ShadeRec& mSr;
};