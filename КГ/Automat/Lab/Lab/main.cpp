#include <iostream>
#include <chrono>
#include <Model.h>
#include <Camera.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> // glm::vec3
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "assimp-vc141-mtd.lib") //need also assimp-vc141-mtd.dll in the main directory

using namespace std;

GLFWwindow* g_window;
int g_vpWidth = 800;
int g_vpHeight = 600;
double g_elapsed = 0.0;

class IScene {
public:
    virtual void draw() = 0;
    virtual void mouse_callback(GLFWwindow* window, double xpos, double ypos) = 0;
    virtual void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) = 0;
};

IScene* g_scene = nullptr;

class MainScene : public IScene {
public:
    MainScene(GLFWwindow* window) {
        m_window = window;
        m_camera = new Camera(glm::vec3(0.0f, 0.0f, 3.0f));
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        buildShader();
        buildCubeModel();
    }

    ~MainScene() {
        delete m_camera;
        delete m_cubeModel;
        delete m_shader;
    }

    void draw() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        processInput(g_elapsed);

        m_angle += g_elapsed * 60.0;
        if (m_angle > 360.0)
            m_angle = m_angle - 360.0;

        glm::mat4 projection = glm::perspective(glm::radians(m_camera->Zoom), (float)g_vpWidth / (float)g_vpHeight, 0.1f, 100.0f);
        glm::mat4 view = m_camera->GetViewMatrix();

        //cube 1 (red)
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(m_angle), glm::vec3(1.0f, 1.0f, 1.0f));
        model = glm::translate(model, glm::vec3(2.0, 0.0, 2.0f));
        model = glm::rotate(model, glm::radians(m_angle), glm::vec3(1.0f, 1.0f, 1.0f));
        model = glm::scale(model, glm::vec3(0.5f, 0.2f, 0.9f)); //cos(glm::radians(m_angle)) * 0.2f
        auto MV_obj = view * model;

        drawCube(MV_obj, projection, glm::vec3(1.0, 0.0, 0.0));

        //cube 2 (blue)
        
        auto MV_bb = glm::mat4(
            glm::vec4(glm::length(MV_obj[0]), 0.0f, 0.0f, 0.0f),
            glm::vec4(0.0f, glm::length(MV_obj[1]), 0.0f, 0.0f),
            glm::vec4(0.0f, 0.0f, glm::length(MV_obj[2]), 0.0f),
            MV_obj[3]
        );

        MV_bb = glm::translate(MV_bb, glm::vec3(5.0, 0.0, 0.0f));
        MV_bb = glm::scale(MV_bb, glm::vec3(1.0f));
        drawCube(MV_bb, projection, glm::vec3(0.0, 0.0, 1.0));
    }

    void mouse_callback(GLFWwindow* window, double xpos, double ypos) override {
        if (xpos || ypos)
            glfwSetCursorPos(window, 0, 0);
        float xoffset = xpos;
        float yoffset = -ypos;
        m_camera->ProcessMouseMovement(xoffset, yoffset);
    }

    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) override {
        m_camera->ProcessMouseScroll(yoffset);
    }

private:
    GLFWwindow* m_window;
    Model* m_cubeModel;
    Shader* m_shader;
    Camera* m_camera;
    float m_angle = 0.0;

    void drawCube(const glm::mat4& MV, const glm::mat4& projection, const glm::vec3& color) {
        m_shader->use();
        m_shader->setMat3("u_normal_mat", glm::transpose(glm::inverse(glm::mat3(MV))));
        m_shader->setMat4("u_mv_mat", MV);
        m_shader->setMat4("u_projection_mat", projection);
        m_shader->setVec3("u_cam_pos", m_camera->Position);
        m_shader->setVec3("u_color", color);
        m_cubeModel->Draw(*m_shader);
    }

    void buildShader() {
        m_shader = new Shader("shaders/view_mode_v.shader", "shaders/view_mode_f.shader");
    }

    void buildCubeModel() {
        m_cubeModel = new Model("models/cube.obj");
    }

    void processInput(float deltaTime)
    {
        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(m_window, true);

        if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS)
            m_camera->ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS)
            m_camera->ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS)
            m_camera->ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS)
            m_camera->ProcessKeyboard(RIGHT, deltaTime);
    }
};

void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    g_vpWidth = width;
    g_vpHeight = height;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    g_scene->mouse_callback(window, xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_scene->mouse_callback(window, xoffset, yoffset);
}

bool initOpenGL()
{
    // Initialize GLFW functions.
    if (!glfwInit())
    {
        cout << "Failed to initialize GLFW" << endl;
        return false;
    }

    // Request OpenGL 3.3 without obsoleted functions.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    // Create window.
    g_window = glfwCreateWindow(g_vpWidth, g_vpHeight, "OpenGL Test", NULL, NULL);
    if (g_window == NULL)
    {
        cout << "Failed to open GLFW window" << endl;
        glfwTerminate();
        return false;
    }

    // Initialize OpenGL context with.
    glfwMakeContextCurrent(g_window);

    // Hide cursor
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(g_window, 0, 0);

    // Set internal GLEW variable to activate OpenGL core profile.
    glewExperimental = true;

    // Initialize GLEW functions.
    if (glewInit() != GLEW_OK)
    {
        cout << "Failed to initialize GLEW" << endl;
        return false;
    }

    // Ensure we can capture the escape key being pressed.
    glfwSetInputMode(g_window, GLFW_STICKY_KEYS, GL_TRUE);

    // Set callbacks
    glfwSetFramebufferSizeCallback(g_window, reshape);
    glfwSetCursorPosCallback(g_window, mouse_callback);
    glfwSetScrollCallback(g_window, scroll_callback);

    g_scene = new MainScene(g_window);
    return true;
}

void tearDownOpenGL()
{
    // Terminate GLFW.
    glfwTerminate();
}

int main()
{
    // Initialize OpenGL
    if (!initOpenGL())
        return -1;

    auto g_callTime = chrono::system_clock::now();

    // Initialize graphical resources.
    bool isOk = true;

    if (isOk)
    {
        // Main loop until window closed or escape pressed.
        while (glfwGetKey(g_window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(g_window) == 0)
        {
            auto callTime = chrono::system_clock::now();
            chrono::duration<double> elapsed = callTime - g_callTime;
            g_elapsed = elapsed.count();
            g_callTime = callTime;

            // Draw scene.
            g_scene->draw(); 

            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();
        }
    }

    // Cleanup graphical resources.
    delete g_scene;

    // Tear down OpenGL.
    tearDownOpenGL();

    return isOk ? 0 : -1;
}