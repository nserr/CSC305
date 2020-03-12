/*
 * CSC 305 Assignment 3
 * Noah Serr
 * V00891494
 */

#version 450 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 colour;

out vec4 clipPosition;
out vec4 lightPosition;
out vec3 vertexColour;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1);
    clipPosition = gl_Position;

    vec3 lightPos = vec3(0.0, 0.5, 0.0);
    lightPosition = projection * view * model * vec4(lightPos, 1);

    vertexColour = colour;
}