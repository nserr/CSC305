/*
 * CSC 305 Assignment 3
 * Noah Serr
 * V00891494
 */

#version 450 core

in vec4 clipPosition;
in vec4 lightPosition;
in vec3 vertexColour;

out vec4 finalColour;

void main()
{
    vec3 ndcPos = clipPosition.xyz / clipPosition.w;
    vec3 dx = dFdx(ndcPos);
    vec3 dy = dFdy(ndcPos);

    vec3 N = normalize(cross(dx, dy));
    N *= sign(N.z);

    vec4 surfaceToLight = lightPosition - clipPosition;

    float brightness = dot(vec4(N, 1.0), surfaceToLight) / (length(surfaceToLight) * length(N));
    brightness = clamp(brightness, 0, 1);

    finalColour = vec4(brightness * (1.0, 1.0, 1.0) * vertexColour.rgb, 1.0);
}