/*
 * CSC 305 Assignment 2
 * Noah Serr
 * V00891494
 */

#include "assignment.hpp"
#include <math.h>
#include <iostream>

using namespace std;

// Shape Functions

Shape::Shape() : mColour{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col) {
    mColour = col;
}

Colour Shape::getColour() const {
    return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material) {
    mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const {
    return mMaterial;
}

// Camera Functions

Camera::Camera() :
    mEye{ 0.0f, 0.0f, 500.0f },
    mLookAt{ 0.0f },
    mUp{ 0.0f, 1.0f, 0.0f },
    mU{ 1.0f, 0.0f, 0.0f },
    mV{ 0.0f, 1.0f, 0.0f },
    mW{ 0.0f, 0.0f, 1.0f }
{}

void Camera::setEye(atlas::math::Point const& eye) {
    mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt) {
    mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up) {
    mUp = up;
}

void Camera::computeUVW() {
    mW = glm::normalize(mEye - mLookAt);
    mU = glm::normalize(glm::cross(mUp, mW));
    mV = glm::cross(mW, mU);

    if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) && mEye.y > mLookAt.y) {
        mU = { 0.0f, 0.0f, 1.0f };
        mV = { 1.0f, 0.0f, 0.0f };
        mW = { 0.0f, 1.0f, 0.0f };
    }

    if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) && mEye.y < mLookAt.y) {
        mU = { 1.0f, 0.0f, 0.0f };
        mV = { 0.0f, 0.0f, 1.0f };
        mW = { 0.0f, -1.0f, 0.0f };
    }
}

// Pinhole Functions

Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance) {
    mDistance = distance;
}

void Pinhole::setZoom(float zoom) {
    mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const {
    const auto dir = p.x * mU + p.y * mV - mDistance * mW;
    return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World> world) const {
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    Point samplePoint{}, pixelPoint{};
    Ray<Vector> ray{};

    ray.o = mEye;
    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ 0 }; r < world->height; r++) {
        for (int c{ 0 }; c < world->width; c++) {
            Colour pixelAverage{ 0,0,0 };

            for (int j{ 0 }; j < world->sampler->getNumSamples(); j++) {
                ShadeRec trace_data{};
                trace_data.world = world;
                trace_data.t = std::numeric_limits<float>::max();
                samplePoint = world->sampler->sampleUnitSquare();
                pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
                pixelPoint.y = r - 0.5f * world->height + samplePoint.y;
                ray.d = rayDirection(pixelPoint);
                bool hit{};

                for (auto const& obj : world->scene) {
                    hit |= obj->hit(ray, trace_data);
                }

                if (hit) {
                    // Out-of-Gamut handling (max-to-one)
                    Colour tmp = trace_data.material->shade(trace_data);
                    float max = std::max(std::max(tmp.r, tmp.g), tmp.b);

                    if (max > 1.0) {
                        tmp /= max;
                    }

                    pixelAverage += tmp;
                }
            }

            world->image.push_back({pixelAverage.r * avg,
                                    pixelAverage.g * avg,
                                    pixelAverage.b * avg});
        }
    }
}

// Sampler Functions

Sampler::Sampler(int numSamples, int numSets) :
    mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 } {

    mSamples.reserve(mNumSets* mNumSamples);
    setupShuffledIndices();
}

int Sampler::getNumSamples() const {
    return mNumSamples;
}

void Sampler::setupShuffledIndices() {
    mShuffledIndices.reserve(mNumSamples * mNumSets);
    std::vector<int> indices;

    std::random_device d;
    std::mt19937 generator(d());

    for (int i{ 0 }; i < mNumSamples; i++) {
        indices.push_back(i);
    }

    for (int j{ 0 }; j < mNumSets; j++) {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int k{ 0 }; k < mNumSamples; k++) {
            mShuffledIndices.push_back(indices[k]);
        }
    }
}

atlas::math::Point Sampler::sampleUnitSquare() {
    if (mCount % mNumSamples == 0) {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return mSamples[mJump + mShuffledIndices[mJump + mCount++ % mNumSamples]];
}

// Light Functions

Colour Light::L([[maybe_unused]] ShadeRec& sr) {
    return mRadiance * mColour;
}

void Light::scaleRadiance(float b) {
    mRadiance = b;
}

void Light::setColour(Colour const& c) {
    mColour = c;
}

// Plane Functions

Plane::Plane(atlas::math::Point p, atlas::math::Normal norm) :
    p_{p}, norm_{norm}
{}

bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    if (intersect && t < sr.t) {
        sr.normal = norm_;
        sr.ray = ray;
        sr.colour = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    const auto o_c{ p_ - ray.o };
    const auto a{ glm::dot(o_c, norm_) };
    const auto b = glm::dot(ray.d, norm_);
    const auto t{ a / b };
    
    if (t > 0.0001f) {
        tmin = t;
        return true;
    }
    
    return false;
}

// Sphere Functions

Sphere::Sphere(atlas::math::Point center, float radius) :
    mCenter{center}, mRadius{radius}, mRadiusSqr{radius * radius}
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
    atlas::math::Vector o_c = ray.o - mCenter;
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    if (intersect && t < sr.t) {
        sr.normal = (o_c + t * ray.d) / mRadius;
        sr.ray = ray;
        sr.colour = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    const auto o_c{ ray.o - mCenter };
    const auto a{ glm::dot(ray.d, ray.d) };
    const auto b{ 2.0f * glm::dot(ray.d, o_c) };
    const auto c{ glm::dot(o_c, o_c) - mRadiusSqr };
    const auto roots{ (b * b) - (4.0f * a * c) };

    if (atlas::core::geq(roots, 0.0f)) {
        const float kEpsilon{ 0.01f };
        const float e{ std::sqrt(roots) };
        const float denom{ 2.0f * a };

        float t = (-b - e) / denom;
        if (atlas::core::geq(t, kEpsilon)) {
            tmin = t;
            return true;
        }

        t = (-b + e);
        if (atlas::core::geq(t, kEpsilon)) {
            tmin = t;
            return true;
        }
    }

    return false;
}

// Triangle Functions

Triangle::Triangle(atlas::math::Point p1, atlas::math::Point p2, atlas::math::Point p3) :
    a_{p1}, b_{p2}, c_{p3}
{}

bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const {
    atlas::math::Vector norm_tmp{ glm::cross((b_ - a_), (c_ - a_)) };
    atlas::math::Normal norm = glm::normalize(norm_tmp);

    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    if (intersect && t < sr.t) {
        sr.normal = norm;
        sr.ray = ray;
        sr.colour = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }

    return intersect;
}

bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    const auto a{ a_.x - b_.x }, b{ a_.x - c_.x }, c{ ray.d.x }, d{ a_.x - ray.o.x };
    const auto e{ a_.y - b_.y }, f{ a_.y - c_.y }, g{ ray.d.y }, h{ a_.y - ray.o.y };
    const auto i{ a_.z - b_.z }, j{ a_.z - c_.z }, k{ ray.d.z }, l{ a_.z - ray.o.z };

    const auto denom = a * (f * k - g * j) + b * (g * i - e * k) + c * (e * j - f * i);

    const auto numBeta = d * (f * k - g * j) - b * (h * k - g * l) - c * (f * l - h * j);
    const auto numGamma = a * (h * k - g * l) + d * (g * i - e * k) + c * (e * l - h * i);

    const auto beta = numBeta / denom;
    const auto gamma = numGamma / denom;
    const auto sum = beta + gamma;

    if (beta < 0) {
        return false;
    }

    if (gamma < 0) {
        return false;
    }

    if (sum > 1) {
        return false;
    }

    const auto numT = a * (f * l - h * j) - b * (e * l - h * i) + d * (e * j - f * i);
    const auto t = numT / denom;

    if (t < 0.0001f) {
        return false;
    }

    tmin = t;
    return true;
}

// Regular Sample Functions

Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets } {
    generateSamples();
}

void Regular::generateSamples() {
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < n; j++) {
            for (int k{ 0 }; k < n; k++) {
                mSamples.push_back(atlas::math::Point{ (k + 0.5f) / n, (j + 0.5f) / n, 0.0f });
            }
        }
    }
}

// Random Sample Functions

Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets } {
    generateSamples();
}

void Random::generateSamples() {
    atlas::math::Random<float> engine;

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < mNumSamples; j++) {
            mSamples.push_back(atlas::math::Point{
                engine.getRandomOne(), engine.getRandomOne(), 0.0f });
        }
    }
}

// Jittered Sample Functions

Jittered::Jittered(int numSamples, int numSets) : Sampler{ numSamples, numSets } {
    generateSamples();
}

void Jittered::generateSamples() {
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
    atlas::math::Random<float> engine;

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < n; j++) {
            for (int k{ 0 }; k < n; k++) {
                mSamples.push_back(atlas::math::Point{ (k + engine.getRandomOne()) / n, (j + engine.getRandomOne()) / n, 0.0f });
            }
        }
    }
}

// Lambertian Functions

Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColour, float diffuseReflection) : 
    mDiffuseColour{diffuseColour}, mDiffuseReflection{diffuseReflection}
{}

Colour Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector const& incoming) const {

    return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}

Colour Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const {

    return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseColour(Colour const& colour) {
    mDiffuseColour = colour;
}

void Lambertian::setDiffuseReflection(float kd) {
    mDiffuseReflection = kd;
}

// Matte Functions

Matte::Matte() :
    Material{},
    mDiffuseBRDF{std::make_shared<Lambertian>()},
    mAmbientBRDF{std::make_shared<Lambertian>()}
{}

Matte::Matte(float kd, float ka, Colour colour) : Matte{} {
    setDiffuseColour(colour);
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
}

void Matte::setDiffuseColour(Colour colour) {
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}

void Matte::setDiffuseReflection(float k) {
    mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k) {
    mAmbientBRDF->setDiffuseReflection(k);
}

Colour Matte::shade(ShadeRec& sr) {
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.o;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; i++) {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f) {
            L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
        }
    }

    return L;
}

// Directional Functions

Directional::Directional() : Light{}
{}

Directional::Directional(atlas::math::Vector const& d) : Light{} {
    setDirection(d);
}

void Directional::setDirection(atlas::math::Vector const& d) {
    mDirection = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr) {
    return mDirection;
}

// Point Functions

Point::Point() : Light{}
{}

Point::Point(atlas::math::Vector const& loc) : Light{} {
    setLocation(loc);
}

void Point::setLocation(atlas::math::Vector const& loc) {
    mLoc = loc;
}

atlas::math::Vector Point::getDirection([[maybe_unused]] ShadeRec& sr) {
    auto tmp = sr.ray.o + (sr.t * sr.ray.d);
    return glm::normalize(mLoc - tmp);
}

// Ambient Functions

Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr) {
    return atlas::math::Vector{ 0.0f };
}

int main()
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };
    world->width = 600;
    world->height = 600;
    world->background = { 0,0,0 };
    world->sampler = std::make_shared<Jittered>(4, 83);

    // Sun
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 0,-50,-700 }, 225.0f));
    world->scene[0]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,0.8,0.16 }));
    world->scene[0]->setColour({ 1,0.8,0.16 });

    // Mercury
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -450,-75,-800 }, 50.0f));
    world->scene[1]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.62,0.6,0.6 }));
    world->scene[1]->setColour({ 0.62,0.6,0.6 });

    // Venus
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -250,-225,-800 }, 60.0f));
    world->scene[2]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.76,0.57,0.22 }));
    world->scene[2]->setColour({ 0.76,0.57,0.22 });

    // Earth
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 0,-300,-800 }, 70.0f));
    world->scene[3]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.12,0.25,0.83 }));
    world->scene[3]->setColour({ 0.12,0.25,0.83 });

    // Mars
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 250,-200,-600 }, 50.0f));
    world->scene[4]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.64,0.39,0.28 }));
    world->scene[4]->setColour({ 0.64,0.39,0.28 });

    // Jupiter
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 375,32,-600 }, 110.0f));
    world->scene[5]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.81,0.78,0.69 }));
    world->scene[5]->setColour({ 0.81,0.78,0.69 });

    // Saturn
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 125,175,-500 }, 95.0f));
    world->scene[6]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.79,0.68,0.46 }));
    world->scene[6]->setColour({ 0.79,0.68,0.46 });

    // Uranus
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -125,175,-500 }, 80.0f));
    world->scene[7]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.78,0.93,0.94 }));
    world->scene[7]->setColour({ 0.78,0.93,0.94 });

    // Neptune
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -300,75,-600 }, 75.0f));
    world->scene[8]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.27,0.45,1 }));
    world->scene[8]->setColour({ 0.27,0.45,1 });
    
    world->scene.push_back(std::make_shared<Triangle>(atlas::math::Point{ -300,-250,-400 }, atlas::math::Point{ -350,-350,-400 }, atlas::math::Point{ -250,-350,-400 }));
    world->scene[9]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,1,1 }));
    world->scene[9]->setColour({ 1,1,1 });

    world->scene.push_back(std::make_shared<Triangle>(atlas::math::Point{ 300,250,-400 }, atlas::math::Point{ 350,350,-400 }, atlas::math::Point{ 250,350,-400 }));
    world->scene[10]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0,0,0 }));
    world->scene[10]->setColour({ 0,0,0 });

    world->scene.push_back(std::make_shared<Plane>(atlas::math::Point{ 0,300,-900 }, Vector{ 0,-1,0 }));
    world->scene[11]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,1,1 }));
    world->scene[11]->setColour({ 1,1,1 });

    world->ambient = std::make_shared<Ambient>();
    world->ambient->setColour({ 1,1,1 });
    world->ambient->scaleRadiance(0.5f);

    world->lights.push_back(std::make_shared<Point>(Point{ { -300,-300,500 } }));
    world->lights[0]->setColour({ 1,1,1 });
    world->lights[0]->scaleRadiance(5.0f);

    Pinhole camera{};
    camera.setEye({ 0.0f, 0.0f, 300.0f });
    camera.computeUVW();
    camera.renderScene(world);

    saveToBMP("C:/Users/noahs/OneDrive/Desktop/School/CSC 305/Assignments/A2/bundle/point.bmp", world->width, world->height, world->image);

    world->image.clear();
    world->lights.clear();
    world->lights.push_back(std::make_shared<Directional>(Directional{ {0,0,1024} }));
    world->lights[0]->setColour({ 1,1,1 });
    world->lights[0]->scaleRadiance(3.0f);
    world->ambient->scaleRadiance(4.0f);
    camera.renderScene(world);

    saveToBMP("C:/Users/noahs/OneDrive/Desktop/School/CSC 305/Assignments/A2/bundle/directional.bmp", world->width, world->height, world->image);
    return 0;
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
