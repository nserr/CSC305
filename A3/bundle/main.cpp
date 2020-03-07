#include "assignment.hpp"
#include "build/_deps/glfw-src/deps/glad/gl.h"

// ===---------------TRIANGLE-----------------===

Triangle::Triangle()
{
    mProgramHandle = glCreateProgram();
    mVertHandle = glCreateShader(GL_VERTEX_SHADER);
    mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
}

void Triangle::loadShaders()
{
    std::string shaderRoot{ ShaderPath };
    vertexSource = glx::readShaderSource(shaderRoot + "triangle.vert", IncludeDir);
    fragmentSource = glx::readShaderSource(shaderRoot + "triangle.frag", IncludeDir);

    if (auto result{ glx::compileShader(vertexSource.sourceString, mVertHandle) }; result)
        throw OpenGLError(*result);

    if (auto result{ glx::compileShader(fragmentSource.sourceString, mFragHandle) }; result)
        throw OpenGLError(*result);

    glAttachShader(mProgramHandle, mVertHandle);
    glAttachShader(mProgramHandle, mFragHandle);

    if (auto result{ glx::linkShaders(mProgramHandle) }; result)
        throw OpenGLError(*result);
}

void Triangle::loadDataToGPU(
    [[maybe_unused]] std::array<float, 18> const& vertices)
{
    glCreateBuffers(1, &mVbo);
    glNamedBufferStorage(mVbo, glx::size<float>(vertices.size()), vertices.data(), 0);

    glCreateVertexArrays(1, &mVao);
    glVertexArrayVertexBuffer(mVao, 0, mVbo, 0, glx::stride<float>(6));

    glEnableVertexArrayAttrib(mVao, 0);
    glEnableVertexArrayAttrib(mVao, 1);

    glVertexArrayAttribFormat(mVao, 0, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(0));
    glVertexArrayAttribFormat(mVao, 1, 3, GL_FLOAT, GL_FALSE, glx::relativeOffset<float>(3));

    glVertexArrayAttribBinding(mVao, 0, 0);
    glVertexArrayAttribBinding(mVao, 1, 0);
}

void Triangle::reloadShaders()
{
    if (glx::shouldShaderBeReloaded(vertexSource))
        glx::reloadShader(mProgramHandle, mVertHandle, vertexSource, IncludeDir);

    if (glx::shouldShaderBeReloaded(fragmentSource))
        glx::reloadShader(mProgramHandle, mFragHandle, fragmentSource, IncludeDir);
}

void Triangle::render()
{
    reloadShaders();
    glUseProgram(mProgramHandle);
    glBindVertexArray(mVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Triangle::freeGPUData()
{
    glDeleteVertexArrays(1, &mVao);
    glDeleteBuffers(1, &mVbo);
    glDeleteShader(mVertHandle);
    glDeleteShader(mFragHandle);
    glDeleteProgram(mProgramHandle);
}

// ===------------IMPLEMENTATIONS-------------===

Program::Program(int width, int height, std::string title) :
    settings{}, callbacks{}, mWindow{ nullptr }
{
    settings.size.width = width;
    settings.size.height = height;
    settings.title = title;

    if (!glx::initializeGLFW(errorCallback))
    {
        throw OpenGLError("Failed to initialize GLFW with error callback");
    }

    mWindow = glx::createGLFWWindow(settings);
    if (mWindow == nullptr)
    {
        throw OpenGLError("Failed to create GLFW Window");
    }

    createGLContext();
}

void Program::run(Triangle& tri)
{
    while (!glfwWindowShouldClose(mWindow))
    {
        int width;
        int height;
        glfwGetFramebufferSize(mWindow, &width, &height);
        // setup the view to be the window's size
        glViewport(0, 0, width, height);
        // tell OpenGL the what color to clear the screen to
        glClearColor(0, 0, 0, 1);
        // actually clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        tri.render();

        glfwSwapBuffers(mWindow);
        glfwPollEvents();
    }
}

void Program::freeGPUData()
{
    glx::destroyGLFWWindow(mWindow);
    glx::terminateGLFW();
}

void Program::createGLContext()
{
    glx::bindWindowCallbacks(mWindow, callbacks);
    glfwMakeContextCurrent(mWindow);
    glfwSwapInterval(1);

    if (!glx::createGLContext(mWindow, settings.version))
    {
        throw OpenGLError("Failed to create OpenGL context");
    }

    glx::initializeGLCallback(
        glx::ErrorSource::All, glx::ErrorType::All, glx::ErrorSeverity::All);
}

// ===-----------------DRIVER-----------------===

int main()
{
    try
    {
        // clang-format off
        std::array<float, 18> vertices =
        {
            // Vertices          Colours
            0.5f, -0.5f, 0.0f,   1.0f, 0.0f, 0.0f,
           -0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,
            0.0f,  0.5f, 0.0f,   0.0f, 0.0f, 1.0f
        };
        // clang-format on

        Program prog{ 1280, 720, "CSC305 Lab 5" };
        Triangle tri{};

        tri.loadShaders();
        tri.loadDataToGPU(vertices);

        prog.run(tri);

        prog.freeGPUData();
        tri.freeGPUData();
    }
    catch (OpenGLError & err)
    {
        fmt::print("OpenGL Error:\n\t{}\n", err.what());
    }

    return 0;
}