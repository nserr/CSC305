/*
 * CSC 305 Assignment 4
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
                    trace_data.world->whitted = std::make_shared<Whitted>(trace_data);
                    pixelAverage += trace_data.material->shade(trace_data);
                }
            }

            world->image.push_back({ pixelAverage.r * avg,
                                    pixelAverage.g * avg,
                                    pixelAverage.b * avg });
        }
    }
}

// Whitted Functions

Whitted::Whitted(ShadeRec& sr) : mSr{ sr }
{}

Colour Whitted::traceRay(atlas::math::Ray<atlas::math::Vector> const& ray, const int depth) const {
    if (depth > mSr.world->maxDepth) {
        return mSr.world->background;
    }
    else {
        bool hit = false;
        for (auto const& obj : mSr.world->scene) {
            hit |= obj->hit(ray, mSr);
        }

        if (hit) {
            mSr.depth = depth;
            mSr.ray = ray;

            return mSr.material->shade(mSr);
        }
        else {
            return mSr.world->background;
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

atlas::math::Point Sampler::sampleHemisphere() {
    if (mCount % mNumSamples == 0) {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return hemisphereSamples[mJump + mShuffledIndices[mJump + mCount++ % mNumSamples]];
}

void Sampler::mapSamplesToHemisphere(const float e) {
    size_t size = mSamples.size();

    hemisphereSamples.reserve(mNumSamples * mNumSets);

    for (size_t i{ 0 }; i < size; i++) {
        float cosPhi = cos(2.0f * glm::pi<float>() * mSamples[i].x);
        float sinPhi = sin(2.0f * glm::pi<float>() * mSamples[i].x);

        float cosTheta = glm::pow((1.0f - mSamples[i].y), 1.0f / (e + 1.0f));
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

        float pu = sinTheta * cosPhi;
        float pv = sinTheta * sinPhi;
        float pw = cosTheta;

        hemisphereSamples.push_back(atlas::math::Point(pu, pv, pw));
    }
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
    p_{ p }, norm_{ norm }
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

bool Plane::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    return intersectRay(ray, tmin);
}

// Sphere Functions

Sphere::Sphere(atlas::math::Point center, float radius) :
    mCenter{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
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

bool Sphere::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    return intersectRay(ray, tmin);
}

// Triangle Functions

Triangle::Triangle(atlas::math::Point p1, atlas::math::Point p2, atlas::math::Point p3) :
    a_{ p1 }, b_{ p2 }, c_{ p3 }
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

bool Triangle::shadowHit(atlas::math::Ray<atlas::math::Vector> const& ray, float& tmin) const {
    return intersectRay(ray, tmin);
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

// MultiJittered Sample Functions

MultiJittered::MultiJittered(int numSamples, int numSets) : Sampler{ numSamples, numSets } {
    generateSamples();
}

void MultiJittered::generateSamples() {
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
    float subcellWidth = 1.0f / static_cast<float>(mNumSamples);
    atlas::math::Random<float> engineF;
    atlas::math::Random<int> engineI;

    atlas::math::Point fillPoint;
    for (int i{ 0 }; i < mNumSamples * mNumSets; i++) {
        mSamples.push_back(fillPoint);
    }

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < n; j++) {
            for (int k{ 0 }; k < n; k++) {
                mSamples[j * n + k + i * mNumSamples].x = (j * n + k) * subcellWidth + engineF.getRandomRange(0, subcellWidth);
                mSamples[j * n + k + i * mNumSamples].y = (k * n + j) * subcellWidth + engineF.getRandomRange(0, subcellWidth);
            }
        }
    }

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < n; j++) {
            for (int k{ 0 }; k < n; k++) {
                int l = engineI.getRandomRange(k, n - 1);
                float t = mSamples[j * n + k + i * mNumSamples].x;
                mSamples[j * n + k + i * mNumSamples].x = mSamples[j * n + l + i * mNumSamples].x;
                mSamples[j * n + l + i * mNumSamples].x = t;
            }
        }
    }

    for (int i{ 0 }; i < mNumSets; i++) {
        for (int j{ 0 }; j < n; j++) {
            for (int k{ 0 }; k < n; k++) {
                int l = engineI.getRandomRange(k, n - 1);
                float t = mSamples[k * n + j + i * mNumSamples].y;
                mSamples[k * n + j + i + mNumSamples].y = mSamples[l * n + j + i * mNumSamples].y;
                mSamples[l * n + j + i * mNumSamples].y = t;
            }
        }
    }
}

// Lambertian Functions

Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColour, float diffuseReflection) :
    mDiffuseColour{ diffuseColour }, mDiffuseReflection{ diffuseReflection }
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

// Perfect Specular Functions

PerfectSpecular::PerfectSpecular(float kr, Colour cr) : BRDF(), mKr{ kr }, mCr{ cr }
{}

void PerfectSpecular::setKr(float kr) {
    mKr = kr;
}

void PerfectSpecular::setCr(Colour cr) {
    mCr = cr;
}

Colour PerfectSpecular::fn([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector const& incoming) const {

    return Colour(0, 0, 0);
}

Colour PerfectSpecular::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const {

    return Colour(0, 0, 0);
}

Colour PerfectSpecular::sampleF(ShadeRec& sr, atlas::math::Vector& wo, atlas::math::Vector& wi) const {
    float nDotWo = glm::dot(sr.normal, wo);
    wi = -wo + 2.0f * sr.normal * nDotWo;
    return (mKr * mCr / glm::dot(sr.normal, wi));
}

// Glossy Specular Functions

GlossySpecular::GlossySpecular() : mKs{}, mCs{}, mExp{}
{}

GlossySpecular::GlossySpecular(float ks, Colour cs, float exp) : mKs{ ks }, mCs{ cs }, mExp{ exp }
{}

Colour GlossySpecular::fn(ShadeRec const& sr,
    atlas::math::Vector const& wo,
    atlas::math::Vector const& wi) const {

    Colour L{ 0,0,0 };

    float nDotWi = glm::dot(sr.normal, wi);
    atlas::math::Vector r(-wi + 2.0f * sr.normal * nDotWi);
    float rDotWo = glm::dot(r, wo);

    if (rDotWo > 0.0f) {
        L = mKs * mCs * glm::pow(rDotWo, mExp);
    }

    return L;
}

Colour GlossySpecular::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const {

    return Colour{ 0,0,0 };
}

void GlossySpecular::setSpecularReflection(float ks) {
    mKs = ks;
}

void GlossySpecular::setSpecularColour(Colour cs) {
    mCs = cs;
}

void GlossySpecular::setExp(float exp) {
    mExp = exp;
}

void GlossySpecular::setSamples(const int numSamples, const int numSets, const float exp) {
    sPtr = std::make_shared<MultiJittered>(numSamples, numSets);
    sPtr->mapSamplesToHemisphere(exp);
}

Colour GlossySpecular::sampleF(ShadeRec& sr,
    atlas::math::Vector& wo,
    atlas::math::Vector& wi,
    float& pdf) const {

    using atlas::math::Vector;

    float nDotWo = glm::dot(sr.normal, wo);
    Vector r = -wo + 2.0f * sr.normal * nDotWo;

    Vector w = r;
    Vector u = glm::cross(Vector(0.00424f, 1.0f, 0.00764f), w);
    glm::normalize(u);
    Vector v = glm::cross(u, w);

    atlas::math::Point sp = sPtr->sampleHemisphere();
    wi = sp.x * u + sp.y * v + sp.z * w;

    if (glm::dot(sr.normal, wi) < 0.0f) {
        wi = -sp.x * u - sp.y * v + sp.z * w;
    }

    float rDotWi = glm::dot(r, wi);
    float phongLobe = glm::pow(rDotWi, mExp);
    pdf = phongLobe * glm::dot(sr.normal, wi);

    return (mKs * mCs * phongLobe);
}

// Perfect Transmitter Functions

PerfectTransmitter::PerfectTransmitter() : mKt{}, mIor{}
{}

PerfectTransmitter::PerfectTransmitter(const float kt, const float ior) : mKt{ kt }, mIor{ ior }
{}

void PerfectTransmitter::setKt(const float k) {
    mKt = k;
}

void PerfectTransmitter::setIor(const float ior) {
    mIor = ior;
}

bool PerfectTransmitter::tir(ShadeRec& sr) const {
    atlas::math::Vector wo(-sr.ray.d);
    float cosThetaI = glm::dot(sr.normal, wo);
    float eta = mIor;

    if (cosThetaI < 0.0f) {
        eta = 1.0f / eta;
    }

    return ((1.0f - (1.0f - cosThetaI * cosThetaI) / (eta * eta)) < 0.0f);
}

Colour PerfectTransmitter::f([[maybe_unused]] ShadeRec& sr,
    [[maybe_unused]] atlas::math::Vector& wo,
    [[maybe_unused]] atlas::math::Vector& wi) const {

    return Colour(0, 0, 0);
}

Colour PerfectTransmitter::sampleF(ShadeRec& sr,
    atlas::math::Vector& wo,
    atlas::math::Vector& wt) const {

    atlas::math::Normal n(sr.normal);
    float cosThetaI = glm::dot(n, wo);
    float eta = mIor;

    if (cosThetaI < 0.0f) {
        cosThetaI = -cosThetaI;
        n = -n;
        eta = 1.0f / eta;
    }

    float tmp = 1.0f - (1.0f - cosThetaI * cosThetaI) / (eta * eta);
    float cosTheta2 = sqrt(tmp);
    wt = -wo / eta - (cosTheta2 - cosThetaI / eta) * n;

    return (mKt / (eta * eta) * Colour(1, 1, 1) / fabs(glm::dot(sr.normal, wt)));
}

Colour PerfectTransmitter::rho([[maybe_unused]] ShadeRec& sr, 
    [[maybe_unused]] atlas::math::Vector& wo) const {
    return Colour(0, 0, 0);
}

// Matte Functions

Matte::Matte() :
    Material{},
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
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
            bool inShadow = false;
            if (sr.world->lights[i]->castsShadows()) {
                Ray<Vector> shadowRay;
                shadowRay.o = sr.ray.o + sr.t * sr.ray.d;
                shadowRay.d = wi;

                inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
            }

            if (!inShadow) {
                L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
            }
        }
    }

    return L;
}

// Phong Functions

Phong::Phong() : Material{}, diffuse{ Lambertian() }, ambient{ Lambertian() }, specular{ GlossySpecular() }
{}

Phong::Phong(float ka, float kd, Colour c, float ks, Colour cs, float exp) : Phong{} {
    setAmbientReflection(ka);
    setDiffuseReflection(kd);
    setDiffuseColour(c);
    setSpecularReflection(ks);
    setSpecularColour(cs);
    setExp(exp);
}

void Phong::setDiffuseReflection(float dr) {
    diffuse.setDiffuseReflection(dr);
}

void Phong::setAmbientReflection(float ar) {
    ambient.setDiffuseReflection(ar);
}

void Phong::setDiffuseColour(Colour c) {
    diffuse.setDiffuseColour(c);
    ambient.setDiffuseColour(c);
}

void Phong::setSpecularReflection(float ks) {
    specular.setSpecularReflection(ks);
}

void Phong::setSpecularColour(Colour cs) {
    specular.setSpecularColour(cs);
}

void Phong::setExp(float exp) {
    specular.setExp(exp);
}

Colour Phong::shade(ShadeRec& sr) {
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.d;
    Colour L = ambient.rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; i++) {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f) {
            bool inShadow = false;
            if (sr.world->lights[i]->castsShadows()) {
                Ray<Vector> shadowRay;
                shadowRay.o = sr.ray.o + sr.t * sr.ray.d;
                shadowRay.d = wi;

                inShadow = sr.world->lights[i]->inShadow(shadowRay, sr);
            }

            if (!inShadow) {
                L += diffuse.fn(sr, wo, wi) + specular.fn(sr, wo, wi) * sr.world->lights[i]->L(sr) * nDotWi;
            }
        }
    }

    return L;
}

// Glossy Reflector Functions

GlossyReflector::GlossyReflector(std::shared_ptr<Phong> phong, std::shared_ptr<GlossySpecular> BRDF) :
    Phong{}, mPhong{ phong }, mBRDF{ BRDF }
{}

void GlossyReflector::setSamples(const int numSamples, const int numSets, const float exp) {
    mBRDF->setSamples(numSamples, numSets, exp);
}

Colour GlossyReflector::shade(ShadeRec& sr) {
    Colour L = Phong::shade(sr);

    atlas::math::Vector wo = -sr.ray.d;
    atlas::math::Vector wi;
    float pdf;

    Colour fr(mBRDF->sampleF(sr, wo, wi, pdf));
    atlas::math::Ray<atlas::math::Vector> reflectedRay;
    reflectedRay.o = sr.ray.o + sr.t * sr.ray.d;
    reflectedRay.d = wi;

    L += fr * sr.world->whitted->traceRay(reflectedRay, sr.depth + 1) * glm::dot(sr.normal, wi) / pdf;
    return L;
}

// Reflective Functions

Reflective::Reflective(std::shared_ptr<Phong> phong, std::shared_ptr<PerfectSpecular> BRDF) :
    Phong{}, mPhong{ phong }, mBRDF{ BRDF }
{}

Colour Reflective::shade(ShadeRec& sr) {
    Colour L = Phong::shade(sr);

    atlas::math::Vector wo = -sr.ray.d;
    atlas::math::Vector wi;

    Colour fr = mBRDF->sampleF(sr, wo, wi);
    atlas::math::Ray<atlas::math::Vector> reflectedRay;
    reflectedRay.o = sr.ray.o + sr.t * sr.ray.d;
    reflectedRay.d = wi;

    L += fr * sr.world->whitted->traceRay(reflectedRay, sr.depth + 1) * glm::dot(sr.normal, wi);
    return L;
}

// Transparent Functions

Transparent::Transparent(std::shared_ptr<Phong> phong, std::shared_ptr<PerfectSpecular> BRDF, std::shared_ptr<PerfectTransmitter> BTDF) : 
    Phong{}, mPhong{ phong }, mBRDF{ BRDF }, mBTDF{ BTDF }
{}

Colour Transparent::shade(ShadeRec& sr) {
    Colour L(Phong::shade(sr));
    atlas::math::Vector wo = -sr.ray.d;
    atlas::math::Vector wi;
    Colour fr = mBRDF->sampleF(sr, wo, wi);
    atlas::math::Ray<atlas::math::Vector> reflectedRay;
    reflectedRay.o = sr.ray.o + sr.t * sr.ray.d;
    reflectedRay.d = wi;

    if (mBTDF->tir(sr)) {
        L += sr.world->whitted->traceRay(reflectedRay, sr.depth + 1);
    }
    else {
        atlas::math::Vector wt;
        Colour ft = mBTDF->sampleF(sr, wo, wt);
        atlas::math::Ray<atlas::math::Vector> transmittedRay;
        transmittedRay.o = sr.ray.o + sr.t * sr.ray.d;
        transmittedRay.d = wt;

        //L += fr * sr.world->whitted->traceRay(reflectedRay, sr.depth + 1) * fabs(glm::dot(sr.normal, wi));
        L += ft * sr.world->whitted->traceRay(transmittedRay, sr.depth + 1) * fabs(glm::dot(sr.normal, wt));
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

bool Directional::castsShadows() {
    return true;
}

bool Directional::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
    [[maybe_unused]] ShadeRec const& sr) const {
    return false;
}

void Directional::setSampler([[maybe_unused]] std::shared_ptr<Sampler> sPtr)
{}

// Point Functions

Point::Point() : Light{}
{}

Point::Point(atlas::math::Vector const& loc) : Light{} {
    setLocation(loc);
}

void Point::setLocation(atlas::math::Vector const& loc) {
    mLoc = loc;
}

atlas::math::Point Point::getLocation() {
    return mLoc;
}

atlas::math::Vector Point::getDirection([[maybe_unused]] ShadeRec& sr) {
    auto tmp = sr.ray.o + (sr.t * sr.ray.d);
    return glm::normalize(mLoc - tmp);
}

bool Point::castsShadows() {
    return true;
}

bool Point::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const {
    float t;
    int numObjects = static_cast<int>(sr.world->scene.size());
    float dX = mLoc.x - ray.o.x;
    float dY = mLoc.y - ray.o.y;
    float dZ = mLoc.z - ray.o.z;
    float d = sqrt((dX * dX) + (dY * dY) + (dZ * dZ));

    for (int i{ 0 }; i < numObjects; i++) {
        if (sr.world->scene[i]->shadowHit(ray, t) && t < d) {
            return true;
        }
    }

    return false;
}

void Point::setSampler([[maybe_unused]] std::shared_ptr<Sampler> sPtr)
{}

// Ambient Functions

Ambient::Ambient() : Light{}
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr) {
    return atlas::math::Vector{ 0.0f };
}

bool Ambient::castsShadows() {
    return false;
}

bool Ambient::inShadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
    [[maybe_unused]] ShadeRec const& sr) const {
    return false;
}

void Ambient::setSampler([[maybe_unused]] std::shared_ptr<Sampler> sPtr)
{}

// Ambient Occluder Functions

AmbientOccluder::AmbientOccluder() : Light{}
{}

void AmbientOccluder::setSampler(std::shared_ptr<Sampler> sPtr) {
    if (mSPtr) {
        mSPtr = NULL;
    }

    mSPtr = sPtr;
    mSPtr->mapSamplesToHemisphere(1.0f);
}

atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr) {
    atlas::math::Point sp = mSPtr->sampleHemisphere();
    return (sp.x * u + sp.y * v + sp.z * w);
}

bool AmbientOccluder::inShadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec const& sr) const {
    float t;
    size_t numObjects = sr.world->scene.size();

    for (size_t i{ 0 }; i < numObjects; i++) {
        if (sr.world->scene[i]->shadowHit(ray, t) && t) {
            return true;
        }
    }

    return false;
}

Colour AmbientOccluder::L(ShadeRec& sr) {
    atlas::math::Vector tmp(0.00001f, 1.0f, 0.00001f);

    w = sr.normal;
    v = glm::cross(w, tmp);
    glm::normalize(v);
    u = glm::cross(v, w);

    atlas::math::Ray<atlas::math::Vector> shadowRay;
    shadowRay.o = sr.ray.o + sr.t * sr.ray.d;
    shadowRay.d = getDirection(sr);
    glm::normalize(shadowRay.d);

    if (inShadow(shadowRay, sr)) {
        return (minAmount * mRadiance * mColour);
    }
    else {
        return (mRadiance * mColour);
    }
}

bool AmbientOccluder::castsShadows() {
    return true;
}

// Out-of-Gamut Check (max-to-one)
void gamutCheck(std::vector<Colour> image) {
    for (int i{ 0 }; i < image.size(); i++) {
        float max = std::max(std::max(image[i].r, image[i].g), image[i].b);
        if (max > 1.0f) {
            image[i].r /= max;
            image[i].g /= max;
            image[i].b /= max;
        }
    }
}

int main()
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };
    world->width = 600;
    world->height = 600;
    world->background = { 0,0,0 };
    world->sampler = std::make_shared<MultiJittered>(4, 83);
    world->maxDepth = 10;

    // Sun
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 0,-50,-700 }, 225.0f));
    std::shared_ptr sunPhongPtr = std::make_shared<Phong>(0.0f, 0.0f, Colour{ 1,1,1 }, 0.5f, Colour{ 1,1,1 }, 2000.0f);
    std::shared_ptr sunBrdfPtr = std::make_shared<PerfectSpecular>(1.0f, Colour{ 1,1,1 });
    std::shared_ptr sunBtdfPtr = std::make_shared<PerfectTransmitter>(0.9f, 1.0f);
    std::shared_ptr sunPtr = std::make_shared<Transparent>(sunPhongPtr, sunBrdfPtr, sunBtdfPtr);
    world->scene[0]->setMaterial(sunPtr);
    world->scene[0]->setColour({ 1,1,1 });

    // Mercury
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -450,-75,-800 }, 50.0f));
    world->scene[1]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.62,0.6,0.6 }));
    world->scene[1]->setColour({ 0.62,0.6,0.6 });

    // Venus
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -250,-225,-800 }, 60.0f));
    world->scene[2]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.76,0.57,0.22 }));
    world->scene[2]->setColour({ 0.76,0.57,0.22 });

    // Earth
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ 0,-325,-800 }, 70.0f));
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
    std::shared_ptr saturnPhongPtr = std::make_shared<Phong>(0.25f, 0.5f, Colour{ 0.79,0.68,0.46 }, 0.75f, Colour{ 1,1,1 }, 100.0f);
    std::shared_ptr saturnBrdfPtr = std::make_shared<PerfectSpecular>(0.75f, Colour{ 1,1,1 });
    std::shared_ptr saturnPtr = std::make_shared<Reflective>(saturnPhongPtr, saturnBrdfPtr);
    world->scene[6]->setMaterial(saturnPtr);
    world->scene[6]->setColour({ 0.79,0.68,0.46 });

    // Uranus
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -125,175,-500 }, 80.0f));
    world->scene[7]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 0.78,0.93,0.94 }));
    world->scene[7]->setColour({ 0.78,0.93,0.94 });

    // Neptune
    world->scene.push_back(std::make_shared<Sphere>(atlas::math::Point{ -300,75,-600 }, 75.0f));
    std::shared_ptr neptunePhongPtr = std::make_shared<Phong>(0.0f, 0.0f, Colour{ 0.27,0.45,1 }, 0.0f, Colour{ 0.27,0.45,1 }, 3000.0f);
    std::shared_ptr neptuneBrdfPtr = std::make_shared<GlossySpecular>(0.9f, Colour{ 0.27,0.45,1 }, 100.0f);
    std::shared_ptr neptunePtr = std::make_shared<GlossyReflector>(neptunePhongPtr, neptuneBrdfPtr);
    neptunePtr->setSamples(100, 83, 3000.0f);
    world->scene[8]->setMaterial(neptunePtr);
    world->scene[8]->setColour({ 0.27,0.45,1 });

    // Left Triangle
    world->scene.push_back(std::make_shared<Triangle>(atlas::math::Point{ -300,-250,-400 }, atlas::math::Point{ -350,-350,-400 }, atlas::math::Point{ -250,-350,-400 }));
    world->scene[9]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,1,1 }));
    world->scene[9]->setColour({ 1,1,1 });

    // Right Triangle
    world->scene.push_back(std::make_shared<Triangle>(atlas::math::Point{ 300,-250,-400 }, atlas::math::Point{ 250,-350,-400 }, atlas::math::Point{ 350,-350,-400 }));
    world->scene[10]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,1,1 }));
    world->scene[10]->setColour({ 1,1,1 });

    // Ground Plane
    world->scene.push_back(std::make_shared<Plane>(atlas::math::Point{ 0,300,-900 }, Vector{ 0,-1,0 }));
    world->scene[11]->setMaterial(std::make_shared<Matte>(0.5f, 0.05f, Colour{ 1,1,1 }));
    world->scene[11]->setColour({ 1,1,1 });

    std::shared_ptr samplerPtr = std::make_shared<MultiJittered>(4, 83);
    world->ambient = std::make_shared<AmbientOccluder>();
    world->ambient->setSampler(samplerPtr);
    world->ambient->setColour({ 1,1,1 });
    world->ambient->scaleRadiance(0.75f);

    world->lights.push_back(std::make_shared<Point>(Point{ { 0,-500,200 } }));
    world->lights[0]->setColour({ 1,1,1 });
    world->lights[0]->scaleRadiance(5.0f);

    Pinhole camera{};
    camera.setEye({ 0.0f, 0.0f, 300.0f });
    camera.computeUVW();
    camera.renderScene(world);
    gamutCheck(world->image);

    saveToBMP("C:/Users/noahs/OneDrive/Desktop/School/CSC 305/Assignments/A4/bundle/render.bmp", world->width, world->height, world->image);

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

    for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
    {
        Colour pixel = image[i];
        data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
        data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
        data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
    }

    stbi_write_bmp(filename.c_str(),
        static_cast<int>(width),
        static_cast<int>(height),
        3,
        data.data());
}