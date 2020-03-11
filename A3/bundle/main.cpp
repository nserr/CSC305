#include "assignment.hpp"
using namespace std;

// ===---------------TRIANGLE-----------------===

Triangle::Triangle()
{
    mProgramHandle = glCreateProgram();
    mVertHandle = glCreateShader(GL_VERTEX_SHADER);
    mFragHandle = glCreateShader(GL_FRAGMENT_SHADER);
    position = 0.0f;
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

    setupUniformVariables();
}

void Triangle::loadDataToGPU(
    [[maybe_unused]] std::array<float, 18*12> const& vertices)
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

void Triangle::render([[maybe_unused]] bool paused,
                      [[maybe_unused]] float xMovement, [[maybe_unused]] float zMovement,
                      [[maybe_unused]] float xLookAt, [[maybe_unused]] float yLookAt,
                      [[maybe_unused]] int width, [[maybe_unused]] int height)
{
    reloadShaders();

    if (paused) {
        position += 1.0f;
    }

    auto modelMatrix{ glm::rotate(math::Matrix4{0.1f},
                      glm::radians(position),
                      math::Vector{0.0f, 1.0f, 0.0f}) };

    auto viewMatrix{ glm::lookAt(
        glm::vec3{xMovement, 0.0f, zMovement},  // Where camera is
        glm::vec3{xLookAt, yLookAt, 0.0f},      // Where camera is looking
        glm::vec3{0.0f, 1.0f, 0.0f}) };         // Camera up vector

    auto projMatrix{ glm::perspective(glm::radians(60.0f),
                                     static_cast<float>(width) / height,
                                     nearVal,
                                     farVal) };

    glUseProgram(mProgramHandle);

    glUniformMatrix4fv(mUniformModelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(mUniformViewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(mUniformProjectionLoc, 1, GL_FALSE, glm::value_ptr(projMatrix));

    glBindVertexArray(mVao);
    glDrawArrays(GL_TRIANGLES, 0, 3*12);
}

void Triangle::freeGPUData()
{
    glDeleteVertexArrays(1, &mVao);
    glDeleteBuffers(1, &mVbo);
    glDeleteShader(mVertHandle);
    glDeleteShader(mFragHandle);
    glDeleteProgram(mProgramHandle);
}

void Triangle::setupUniformVariables()
{
    mUniformModelLoc = glGetUniformLocation(mProgramHandle, "model");
    mUniformViewLoc = glGetUniformLocation(mProgramHandle, "view");
    mUniformProjectionLoc = glGetUniformLocation(mProgramHandle, "projection");
}

// ===------------IMPLEMENTATIONS-------------===

Program::Program(int width, int height, std::string title) :
    settings{}, callbacks{}, paused{}, mWindow{ nullptr },
    xMovement{ 0.0f }, zMovement{ 2.0f }, xLookAt{ 0.0f }, yLookAt{ 0.0f }
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

    callbacks.keyPressCallback = [&](int key, int, int action, int) {
        // Rotation movement (space)
        if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE) {
            paused = !paused;
        }

        // Camera position movement (wasd)
        if (key == GLFW_KEY_W && action == GLFW_RELEASE) {
            zMovement -= 0.1f;
        }
        if (key == GLFW_KEY_S && action == GLFW_RELEASE) {
            zMovement += 0.1f;
        }
        if (key == GLFW_KEY_A && action == GLFW_RELEASE) {
            xMovement -= 0.1f;
        }
        if (key == GLFW_KEY_D && action == GLFW_RELEASE) {
            xMovement += 0.1f;
        }

        // Camera lookAt movement (arrows)
        if (key == GLFW_KEY_UP && action == GLFW_RELEASE) {
            yLookAt += 0.1f;
        }
        if (key == GLFW_KEY_DOWN && action == GLFW_RELEASE) {
            yLookAt -= 0.1f;
        }
        if (key == GLFW_KEY_LEFT && action == GLFW_RELEASE) {
            xLookAt -= 0.1f;
        }
        if (key == GLFW_KEY_RIGHT && action == GLFW_RELEASE) {
            xLookAt += 0.1f;
        }
    };

    createGLContext();
}

void Program::run(Triangle& tri)
{
    glEnable(GL_DEPTH_TEST);

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
        // set cursor to centre of window
        //glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        tri.render(paused, xMovement, zMovement, xLookAt, yLookAt, width, height);

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
        std::array<float, 18*12> vertices =
        {
            // Vertices          Colours
           -0.5f, -0.5f, -0.5f,  0.583f, 0.771f, 0.014f,
           -0.5f, -0.5f,  0.5f,  0.609f, 0.115f, 0.436f,
           -0.5f,  0.5f,  0.5f,  0.327f, 0.483f, 0.844f,

            0.5f,  0.5f, -0.5f,  0.822f, 0.569f, 0.201f,
           -0.5f, -0.5f, -0.5f,  0.435f, 0.602f, 0.223f,
           -0.5f,  0.5f, -0.5f,  0.310f, 0.747f, 0.185f,

            0.5f, -0.5f,  0.5f,  0.597f, 0.770f, 0.761f,
           -0.5f, -0.5f, -0.5f,  0.559f, 0.436f, 0.730f,
            0.5f, -0.5f, -0.5f,  0.359f, 0.583f, 0.152f,

            0.5f,  0.5f, -0.5f,  0.483f, 0.596f, 0.789f,
            0.5f, -0.5f, -0.5f,  0.559f, 0.861f, 0.639f,
           -0.5f, -0.5f, -0.5f,  0.195f, 0.548f, 0.859f,

           -0.5f, -0.5f, -0.5f,  0.014f, 0.184f, 0.576f,
           -0.5f,  0.5f,  0.5f,  0.771f, 0.328f, 0.970f,
           -0.5f,  0.5f, -0.5f,  0.406f, 0.615f, 0.116f,

            0.5f, -0.5f,  0.5f,  0.676f, 0.977f, 0.133f,
           -0.5f, -0.5f,  0.5f,  0.971f, 0.572f, 0.833f,
           -0.5f, -0.5f, -0.5f,  0.140f, 0.616f, 0.489f,

           -0.5f,  0.5f,  0.5f,  0.997f, 0.513f, 0.064f,
           -0.5f, -0.5f,  0.5f,  0.945f, 0.719f, 0.592f,
            0.5f, -0.5f,  0.5f,  0.543f, 0.021f, 0.978f,

            0.5f,  0.5f,  0.5f,  0.279f, 0.317f, 0.505f,
            0.5f, -0.5f, -0.5f,  0.167f, 0.620f, 0.077f,
            0.5f,  0.5f, -0.5f,  0.347f, 0.857f, 0.137f,

            0.5f, -0.5f, -0.5f,  0.055f, 0.953f, 0.042f,
            0.5f,  0.5f,  0.5f,  0.714f, 0.505f, 0.345f,
            0.5f, -0.5f,  0.5f,  0.783f, 0.290f, 0.734f,

            0.5f,  0.5f,  0.5f,  0.722f, 0.645f, 0.174f,
            0.5f,  0.5f, -0.5f,  0.302f, 0.455f, 0.848f,
           -0.5f,  0.5f, -0.5f,  0.225f, 0.587f, 0.040f,

            0.5f,  0.5f,  0.5f,  0.517f, 0.713f, 0.338f,
           -0.5f,  0.5f, -0.5f,  0.053f, 0.959f, 0.120f,
           -0.5f,  0.5f,  0.5f,  0.393f, 0.621f, 0.362f,

            0.5f,  0.5f,  0.5f,  0.673f, 0.211f, 0.457f,
           -0.5f,  0.5f,  0.5f,  0.820f, 0.883f, 0.371f,
            0.5f, -0.5f,  0.5f,  0.982f, 0.099f, 0.879f
        };

        // clang-format on

        Program prog{ 1280, 720, "CSC 305 Assignment 3" };
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