#include <iostream>
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
        ""
        "out vec3 v_normal;"
        "out vec3 v_pos;"
        ""
        "float f(vec2 p) { return sin(p.x) * cos(p.y); }"
        "vec3 grad(vec2 p) { return vec3(-cos(p.x) * cos(p.y), 1.0, sin(p.x) * sin(p.y)); };"
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
        "   float d = max(dot(n, -l), 0.1);"
        "   vec3 e = normalize(E - v_pos);"
        "   vec3 h = normalize(-l + e);"
        "   float s = pow(max(dot(n, h), 0.0), S);"
        "   o_color = vec4(color * d + s * vec3(1, 1, 1), 1);"
        "}"
        ;

    const GLchar vsh2[] =
        "#version 330\n"
        ""
        "layout(location = 0) in vec3 a_position;"
        "layout(location = 1) in vec3 a_color;"
        ""
        "uniform mat4 u_mvp;"
        ""
        "out vec3 v_color;"
        ""
        "void main()"
        "{"
        "    v_color = a_color;"
        "    gl_Position = u_mvp * vec4(a_position, 1.0);"
        "}"
        ;

    const GLchar fsh2[] =
        "#version 330\n"
        ""
        "in vec3 v_color;"
        ""
        "layout(location = 0) out vec4 o_color;"
        ""
        "void main()"
        "{"
        "   o_color = vec4(v_color, 1.0);"
        "}"
        ;

    GLuint vertexShader, fragmentShader;

    vertexShader = createShader(vsh, GL_VERTEX_SHADER);
    fragmentShader = createShader(fsh, GL_FRAGMENT_SHADER);

    g_shaderProgram = createProgram(vertexShader, fragmentShader);

    g_uMVP = glGetUniformLocation(g_shaderProgram, "u_mvp");
    g_uMV = glGetUniformLocation(g_shaderProgram, "u_mv");

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
    GLuint idx_tpl[6] = { 0, 1, 2, 2, 3, 0 };

    for (size_t i = 0; i < N + 1; i++) {
        for (size_t j = 0; j < N + 1; j++) {
            auto vertex = &vertices[(i * (N + 1) + j) * 2];
            vertex[0] = GLfloat(i) / (N + 1) - 0.5f;
            vertex[1] = GLfloat(j) / (N + 1) - 0.5f;
        }
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            auto idx = &indices[(i * N + j) * 6];
            idx[0] = i * N + j;
            idx[1] = i * N + (j + 1);
            idx[2] = (i + 1) * N + (j + 1);
            idx[3] = (i + 1) * N + (j + 1);
            idx[4] = (i + 1) * N + j;
            idx[5] = i * N + j;
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

    return g_model.vbo != 0 && g_model.ibo != 0 && g_model.vao != 0;
}

bool createModel()
{
    const GLfloat vertices[] =
    {
        -1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        -1.0, 1.0, 1.0, 1.0, 0.0, 0.0,

        1.0, -1.0, 1.0, 1.0, 1.0, 0.0,
        1.0, -1.0, -1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, -1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0,

        1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
        -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
        -1.0, 1.0, 1.0, 1.0, 0.0, 1.0,

        -1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        -1.0, 1.0, -1.0, 0.0, 1.0, 1.0,
        -1.0, -1.0, -1.0, 0.0, 1.0, 1.0,
        -1.0, -1.0, 1.0, 0.0, 1.0, 1.0,

        -1.0, -1.0, 1.0, 0.0, 1.0, 0.0,
        -1.0, -1.0, -1.0, 0.0, 1.0, 0.0,
        1.0, -1.0, -1.0, 0.0, 1.0, 0.0,
        1.0, -1.0, 1.0, 0.0, 1.0, 0.0,

        -1.0, -1.0, -1.0, 0.0, 0.0, 1.0,
        -1.0, 1.0, -1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, -1.0, 0.0, 0.0, 1.0,
        1.0, -1.0, -1.0, 0.0, 0.0, 1.0,
    };

    const GLuint indices[] =
    {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        8, 9, 10, 10, 11, 8,
        12, 13, 14, 14, 15, 12,
        16, 17, 18, 18, 19, 16,
        20, 21, 22, 22, 23, 20
    };

    glGenVertexArrays(1, &g_model.vao);
    glBindVertexArray(g_model.vao);

    glGenBuffers(1, &g_model.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_model.vbo);
    glBufferData(GL_ARRAY_BUFFER, 24 * 6 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &g_model.ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_model.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * 6 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    g_model.indexCount = 6 * 6;

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (const GLvoid*)(3 * sizeof(GLfloat)));

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

GLfloat step = 0.0;
void draw()
{
    // Clear color buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(g_shaderProgram);
    glBindVertexArray(g_model.vao);

    glm::mat4 Model = glm::mat4(1.0f);
    Model = glm::translate(Model, glm::vec3(0.0, -4.0, -10.0f));
    Model = glm::rotate(Model, glm::radians(step), glm::vec3(0.0f, 1.0f, 0.0f));
    Model = glm::scale(Model, glm::vec3(2.0f));

    glm::mat4 View = glm::mat4(1.0f);
    View = glm::rotate(View, glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.f);

    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, glm::value_ptr(Projection * View * Model));
    glUniformMatrix4fv(g_uMV, 1, GL_FALSE, glm::value_ptr(View * Model));
    
    glDrawElements(GL_TRIANGLES, g_model.indexCount, GL_UNSIGNED_INT, NULL);
    step += 0.5;
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
        // Main loop until window closed or escape pressed.
        while (glfwGetKey(g_window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(g_window) == 0)
        {
            // Draw scene.
            draw();

            // Swap buffers.
            glfwSwapBuffers(g_window);
            // Poll window events.
            glfwPollEvents();
        }
    }

    // Cleanup graphical resources.
    cleanup();

    // Tear down OpenGL.
    tearDownOpenGL();

    return isOk ? 0 : -1;
}
