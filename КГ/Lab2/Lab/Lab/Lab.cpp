#include <iostream>
#include <chrono>
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

GLuint g_shaderProgram;
GLint g_uMV;
GLint g_uMVP;
GLint g_uA;
GLint g_uB;

class Model
{
public:
    GLuint vbo;
    GLuint ibo;
    GLuint vao;
    GLsizei indexCount;
};

Model g_model;

GLuint createShader(const GLchar* code, GLenum type)
{
    GLuint result = glCreateShader(type);

    glShaderSource(result, 1, &code, NULL);
    glCompileShader(result);

    GLint compiled;
    glGetShaderiv(result, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetShaderInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader compilation error" << endl << infoLog << endl;
        }
        glDeleteShader(result);
        return 0;
    }

    return result;
}

GLuint createProgram(GLuint vsh, GLuint fsh)
{
    GLuint result = glCreateProgram();

    glAttachShader(result, vsh);
    glAttachShader(result, fsh);

    glLinkProgram(result);

    GLint linked;
    glGetProgramiv(result, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        GLint infoLen = 0;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            char* infoLog = (char*)alloca(infoLen);
            glGetProgramInfoLog(result, infoLen, NULL, infoLog);
            cout << "Shader program linking error" << endl << infoLog << endl;
        }
        glDeleteProgram(result);
        return 0;
    }

    return result;
}

bool createShaderProgram()
{
    g_shaderProgram = 0;

    const GLchar vsh[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec2 a_position;"
        ""
        "uniform mat4 u_mvp;"
        "uniform mat4 u_mv;"
        "uniform float u_a;"
        "uniform float u_b;"
        ""
        "out vec3 v_normal;"
        "out vec3 v_pos;"
        ""

        //"float f(vec2 p) { return sin(p.x) * cos(p.y); }"
        //"vec3 grad(vec2 p) { return vec3(-cos(p.x) * cos(p.y), 1.0, sin(p.x) * sin(p.y)); };"

        //"float f(vec2 p) { return 1.5 * atan(2.0 * p.x * p.y); }"
        //"vec3 grad(vec2 p) { return vec3(-3*p.y/(4*p.x*p.x * p.y*p.y + 1), 1.0, -3*p.x/(4*p.x*p.x * p.y*p.y + 1)); };"

        "float f(vec2 p) { return u_a * atan(u_b * p.x * p.y); }"
        "vec3 grad(vec2 p) { return vec3(-u_a*u_b*p.y/(u_b*u_b*p.x*p.x * p.y*p.y + 1), 1.0, -u_a*u_b*p.x/(u_b*u_b*p.x*p.x * p.y*p.y + 1)); };"
        //"float f(vec2 p) { return 0.0; }"
        //"vec3 grad(vec2 p) { return vec3(0, 1, 0); };"

        //"float f(vec2 p) { return 1.5 * (1 - p.x * p.y) * sin(1 - p.x * p.y); }"
        //"vec3 grad(vec2 p) { return vec3(p.y * (1.5 - 1.5 * p.x * p.y) * cos(1 - p.x * p.y) + 1.5 * z * sin(1 - p.x * p.y), 1, p.x * (1.5 - 1.5 * p.x * p.y) * cos(1 - p.x * p.y) + 1.5 * x * sin(1 - p.x * p.y)); };"

        ""
        "void main()"
        "{"
        "   float y = f(a_position);"
        "   vec4 p0 = vec4(a_position[0], y, a_position[1], 1.0);"
        "   v_normal = transpose(inverse(mat3(u_mv))) * normalize(grad(a_position));"
        "   v_pos = vec3(u_mv * p0);"
        "   gl_Position = u_mvp * vec4(a_position[0], y, a_position[1], 1.0);"
        "}"
        ;

    const GLchar fsh[] =
        "#version 330\n"
        ""
        ""
        "in vec3 v_normal;"
        "in vec3 v_pos;"
        ""
        "layout(location = 0) out vec4 o_color;"
        ""
        "void main()"
        "{"
        "   float S = 10;"
        "   vec3 color = vec3(1, 0, 0);"
        "   vec3 n = normalize(v_normal);"
        "   vec3 E = vec3(0, 0, 0);"
        "   vec3 L = vec3(5, 5, 0);"
        "   vec3 l = normalize(v_pos - L);"
        "   float d = max(dot(n, -l), 0.3);"
        "   vec3 e = normalize(E - v_pos);"
        "   vec3 h = normalize(-l + e);"
        "   float s = pow(max(dot(n, h), 0.0), S);"
        "   o_color = vec4(color * d + s * vec3(1, 1, 1), 1);"
        "}"
        ;
    
    GLuint vertexShader, fragmentShader;

    vertexShader = createShader(vsh, GL_VERTEX_SHADER);
    fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER);

    g_shaderProgram = createProgram(vertexShader, fragmentShader);

    g_uMVP = glGetUniformLocation(g_shaderProgram, "u_mvp");
    g_uMV = glGetUniformLocation(g_shaderProgram, "u_mv");
    g_uA = glGetUniformLocation(g_shaderProgram, "u_a");
    g_uB = glGetUniformLocation(g_shaderProgram, "u_b");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return g_shaderProgram != 0;
}

bool createGrid(size_t N = 100)
{
    size_t verticesCount = (N + 1) * (N + 1);
    size_t indicesCount = N * N * 6;

    auto vertices = new GLfloat[verticesCount * 2];
    auto indices = new GLuint[indicesCount];
    
    for (size_t i = 0; i < N + 1; i++) {
        for (size_t j = 0; j < N + 1; j++) {
            auto vertex = &vertices[(i * (N + 1) + j) * 2];
            vertex[0] = GLfloat(j) / N - 0.5f;
            vertex[1] = GLfloat(i) / N - 0.5f;
        }
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            auto idx = &indices[(i * N + j) * 6];
            idx[0] = i * (N + 1) + j;
            idx[1] = i * (N + 1) + (j + 1);
            idx[2] = (i + 1) * (N + 1) + (j + 1);
            idx[3] = (i + 1) * (N + 1) + (j + 1);
            idx[4] = (i + 1) * (N + 1) + j;
            idx[5] = i * (N + 1) + j;
        }
    }

    glGenVertexArrays(1, &g_model.vao);
    glBindVertexArray(g_model.vao);

    glGenBuffers(1, &g_model.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_model.vbo);
    glBufferData(GL_ARRAY_BUFFER, verticesCount * 2 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &g_model.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_model.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesCount * sizeof(GLuint), indices, GL_STATIC_DRAW);

    g_model.indexCount = indicesCount;

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (const GLvoid*)0);

    delete[] vertices;
    delete[] indices;

    return g_model.vbo != 0 && g_model.ibo != 0 && g_model.vao != 0;
}

bool init()
{
    // Set initial color of color buffer to white.
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glEnable(GL_DEPTH_TEST);

    return createShaderProgram() && createGrid();
}

void reshape(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void draw(GLfloat p)
{
    GLfloat model_rotation_angle = p / 8.0;
    GLfloat model_animation_period = p / 4.0;

    // Clear color buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_shaderProgram);
    glBindVertexArray(g_model.vao);

    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::translate(Model, glm::vec3(0.0, -4.0, -10.0f));
    Model = glm::rotate(Model, model_rotation_angle, glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::scale(Model, glm::vec3(5.0f));

    glm::mat4 View = glm::mat4(1.0f);
    View = glm::rotate(View, glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.f);

    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, glm::value_ptr(Projection * View * Model));
    glUniformMatrix4fv(g_uMV, 1, GL_FALSE, glm::value_ptr(View * Model));

    glUniform1f(g_uA, 1.5);
    glUniform1f(g_uB, abs(cos(model_animation_period)) + 0.5);
    
    glDrawElements(GL_TRIANGLES, g_model.indexCount, GL_UNSIGNED_INT, NULL);
}

void cleanup()
{
    if (g_shaderProgram != 0)
        glDeleteProgram(g_shaderProgram);
    if (g_model.vbo != 0)
        glDeleteBuffers(1, &g_model.vbo);
    if (g_model.ibo != 0)
        glDeleteBuffers(1, &g_model.ibo);
    if (g_model.vao != 0)
        glDeleteVertexArrays(1, &g_model.vao);
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

    // Create window.
    g_window = glfwCreateWindow(800, 600, "OpenGL Test", NULL, NULL);
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
    bool isOk = init();

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
            draw(p);

            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();

            auto t2 = std::chrono::high_resolution_clock::now();
            time_to_render_frame = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        }
    }

    // Cleanup graphical resources.
    cleanup();

    // Tear down OpenGL.
    tearDownOpenGL();

    return isOk ? 0 : -1;
}
