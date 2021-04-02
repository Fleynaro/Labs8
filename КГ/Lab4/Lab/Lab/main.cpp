#include <iostream>
#include <chrono>
#include <Mesh.h>
#include <tbezier/tbezier.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp> // glm::vec3
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtc/type_ptr.hpp>
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

using namespace std;

GLFWwindow* g_window;
int g_vpWidth = 800;
int g_vpHeight = 600;

class IScene {
public:
    virtual void draw() = 0;
    virtual void mouse_callback(GLFWwindow* window, int button, int action, int mods) = 0;
};

class EditModeScene : public IScene {
public:
    EditModeScene() {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        buildPointMesh();
        buildShader();
    }

    ~EditModeScene() {
        delete m_pointMesh;
        if (m_curveMesh) {
            delete m_curveMesh;
        }
        delete m_shader;
    }

    void draw() override {
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_MULTISAMPLE);

        auto projection = glm::ortho(0.0f, float(g_vpWidth), 0.0f, float(g_vpHeight), 0.1f, 100.0f);
        for (const auto& point : m_editPoints) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(point.x, point.y, -1.0f));
            model = glm::scale(model, glm::vec3(10.0f));

            m_shader->use();
            m_shader->setVec3("u_color", glm::vec3(1.0f, 0.0f, 0.0f));
            m_shader->setMat4("u_mvp", projection * model);
            m_pointMesh->Draw(*m_shader);
        }

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(g_vpWidth, 5.0f, 1.0f));
        model = glm::translate(model, glm::vec3(0.5, 0.5, -1.0f)); //(model_matrix * scale_matrix) * vertex
        m_shader->use();
        m_shader->setVec3("u_color", glm::vec3(0.0f, 0.0f, 0.5f));
        m_shader->setMat4("u_mvp", projection * model);
        m_pointMesh->Draw(*m_shader);

        if (m_curveMesh) {
            glEnable(GL_MULTISAMPLE);
            m_shader->use();
            m_shader->setVec3("u_color", glm::vec3(0.7f, 0.0f, 0.0f));
            m_shader->setMat4("u_mvp", projection);
            m_curveMesh->Draw(*m_shader);
        }
    }

    void mouse_callback(GLFWwindow* window, int button, int action, int mods) override {
        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT)
        {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            m_editPoints.push_back(Point2D(xpos, g_vpHeight - ypos));
            calculate();
            if (!m_calcPoints.empty())
                buildCurveMesh();
        }
    }

private:
    Mesh* m_pointMesh;
    Mesh* m_curveMesh = nullptr;
    Shader* m_shader;
    vector<Point2D> m_editPoints;
    vector<Point2D> m_calcPoints;
    float m_curveWidth = 2.0f;
    
    void buildShader() {
        m_shader = new Shader("shaders/edit_mode_v.shader", "shaders/edit_mode_f.shader");
    }

    void buildPointMesh() {
        Vertex v1;
        Vertex v2;
        Vertex v3;
        Vertex v4;
        v1.Position = glm::vec3(-0.5f, -0.5f, 0.0f);
        v2.Position = glm::vec3(0.5f, -0.5f, 0.0f);
        v3.Position = glm::vec3(0.5f, 0.5f, 0.0f);
        v4.Position = glm::vec3(-0.5f, 0.5f, 0.0f);

        vector<unsigned int> indices = {
            0, 1, 2, 2, 3, 0
        };

        m_pointMesh = new Mesh({ v1, v2, v3, v4 }, indices, {});
    }

    void buildCurveMesh() {
        Vertex v;
        vector<Vertex> vertices;

        v.Position = glm::vec3(m_calcPoints[0].x, m_calcPoints[0].y, 0.0f);
        vertices.push_back(v);
        vertices.push_back(v);

        for (size_t i = 1; i < m_calcPoints.size() - 1; i++) {
            auto vec1 = m_calcPoints[i] - m_calcPoints[i - 1];
            auto vec2 = m_calcPoints[i + 1] - m_calcPoints[i];
            vec1.normalize();
            vec2.normalize();
            auto h = vec1 + vec2;
            h.normalize();
            //rotate on 90 degrees
            swap(h.x, h.y);
            h.x *= -1.0f;

            auto p1 = m_calcPoints[i] - h * m_curveWidth;
            auto p2 = m_calcPoints[i] + h * m_curveWidth;
            
            v.Position = glm::vec3(p1.x, p1.y, -1.0f);
            vertices.push_back(v);
            v.Position = glm::vec3(p2.x, p2.y, -1.0f);
            vertices.push_back(v);
        }

        v.Position = glm::vec3(m_calcPoints[m_calcPoints.size() - 1].x, m_calcPoints[m_calcPoints.size() - 1].y, 0.0f);
        vertices.push_back(v);
        vertices.push_back(v);

        vector<unsigned int> indices;
        for (size_t i = 0; i < m_calcPoints.size() - 1; i++) {
            indices.push_back(0 + i * 2);
            indices.push_back(1 + i * 2);
            indices.push_back(3 + i * 2);
            indices.push_back(3 + i * 2);
            indices.push_back(2 + i * 2);
            indices.push_back(0 + i * 2);
        }

        if (m_curveMesh) {
            delete m_curveMesh;
        }
        m_curveMesh = new Mesh(vertices, indices, {});
    }

    void calculate() {
        vector<Segment> curve;
        tbezierSO0(m_editPoints, curve);
        m_calcPoints.clear();

        for (auto s : curve)
        {
            for (int i = 0; i < RESOLUTION; ++i)
            {
                Point2D p = s.calc((double)i / (double)RESOLUTION);
                m_calcPoints.push_back(p);
            }
        }
    }
};

IScene* g_scene = nullptr;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    g_scene->mouse_callback(window, button, action, mods);

    if (action == GLFW_PRESS) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {

        }
    }
}

void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    g_vpWidth = width;
    g_vpHeight = height;
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

    // Set callback for framebuffer resizing event.
    glfwSetFramebufferSizeCallback(g_window, reshape);
    glfwSetMouseButtonCallback(g_window, mouse_button_callback);
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

    // Initialize graphical resources.
    g_scene = new EditModeScene();
    bool isOk = true;

    if (isOk)
    {
        GLfloat p = 0.0;
        const GLfloat delta_p = glm::pi<GLfloat>();
        long long time_to_render_frame = 0.0;

        // Main loop until window closed or escape pressed.
        while (glfwGetKey(g_window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(g_window) == 0)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            //Animation.
            p += time_to_render_frame * delta_p / 1000.0f;
            if (p > 1024 * glm::pi<GLfloat>()) {
                p -= 1024 * glm::pi<GLfloat>();
            }

            // Draw scene.
            g_scene->draw();
            

            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();

            auto t2 = std::chrono::high_resolution_clock::now();
            time_to_render_frame = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
    }

    // Cleanup graphical resources.
    delete g_scene;

    // Tear down OpenGL.
    tearDownOpenGL();

    return isOk ? 0 : -1;
}