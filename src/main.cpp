#include <iostream>
#include <cmath>
#include <string>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <mpi.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

enum RenderMode : uint8_t { LocalDisplay, ImageCapture };

typedef struct LocalViewport {
    int column;
    int row;
    int width;
    int height;
    int num_columns;
    int num_rows;
} LocalViewport;

typedef struct GShaderProgram {
    GLuint program;
    GLint proj_uniform;
    GLint mv_uniform;
    GLint norm_uniform;
    GLint ambientcol_uniform;
    GLint lightdir_uniform;
    GLint lightcol_uniform;
    GLint img_uniform;
} GShaderProgram;

typedef struct AppData {
    int rank;
    int num_ranks;
    RenderMode render_mode;
    GLuint vao;
    GLuint tex_id;
    GLuint vertex_position_attrib;
    GLuint vertex_normal_attrib;
    GLuint vertex_texcoord_attrib;
    glm::mat4 mat_projection;
    glm::mat4 mat_modelview;
    double render_time;
    double rotate_x;
    double rotate_y;
    int frame_count;
    uint8_t *framebuffer;
} AppData;

static void Init(GLFWwindow *window, GShaderProgram *shader, AppData *app, LocalViewport& viewport);
static void Idle(GLFWwindow *window, GShaderProgram& shader, AppData& app, LocalViewport& viewport);
static void Render(GLFWwindow *window, GShaderProgram& shader, AppData& app, LocalViewport& viewport);
static void SetMatrixUniforms(GShaderProgram& shader, AppData& app);
static GLuint CreateCubeVao(AppData& app);
static GShaderProgram CreateTextureShader(AppData& app);
static GLint CompileShader(char *source, uint32_t length, GLint type);
static void CreateShaderProgram(GLint vertex_shader, GLint fragment_shader, GLuint *program);
static void LinkShaderProgram(GLuint program);
static int32_t ReadFile(const char* filename, char** data_ptr);
static void GetClosestFactors2(int value, int *factor_1, int *factor_2);

int main(int argc, char **argv)
{
    // initialize MPI
    int rc, rank, num_ranks;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (rc != 0)
    {
        fprintf(stderr, "Error initializing MPI and obtaining task ID information\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // read command line parameters for overall width / height
    AppData app;
    app.rank = rank;
    app.num_ranks = num_ranks;
    app.render_mode = RenderMode::LocalDisplay;
    int width = 1280;
    int height = 720;
    if (argc >= 2 && std::string(argv[1]) == "imagecapture") app.render_mode = RenderMode::ImageCapture;
    if (argc >= 3) width = atoi(argv[2]);
    if (argc >= 4) height = atoi(argv[3]);

    // initialize GLFW
    if (!glfwInit())
    {
        exit(1);
    }

    // calculate window size and position
    int rows, cols;
    GetClosestFactors2(num_ranks, &cols, &rows);
    LocalViewport m_viewport;
    m_viewport.column = rank % cols;
    m_viewport.row = rank / cols;
    m_viewport.width = width / cols;
    m_viewport.height = height / rows;
    m_viewport.num_columns = cols;
    m_viewport.num_rows = rows;

    // create a window and its OpenGL context
    char title[32];
    snprintf(title, 32, "Texture Cube: %d", rank);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *window = glfwCreateWindow(m_viewport.width, m_viewport.height, title, NULL, NULL);

    // make window's context current
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize GLAD OpenGL extension handling
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        exit(1);
    }

    // initialize app
    GShaderProgram shader;
    Init(window, &shader, &app, m_viewport);

    // main render loop
    Render(window, shader, app, m_viewport); 
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        Idle(window, shader, app, m_viewport);
    }

    // clean up
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void Init(GLFWwindow *window, GShaderProgram *shader, AppData *app, LocalViewport& viewport)
{
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.9, 0.9, 0.9, 1.0);
    glEnable(GL_DEPTH_TEST);

    app->framebuffer = new uint8_t[w * h * 4];
    app->vertex_position_attrib = 0;
    app->vertex_normal_attrib = 1;
    app->vertex_texcoord_attrib = 2;
    app->frame_count = 0;

    *shader = CreateTextureShader(*app);
    app->vao = CreateCubeVao(*app);

    glGenTextures(1, &(app->tex_id));
    glBindTexture(GL_TEXTURE_2D, app->tex_id);
    int img_w, img_h, img_c;
    stbi_set_flip_vertically_on_load(true);
    uint8_t *pixels = stbi_load("resrc/images/crate.jpg", &img_w, &img_h, &img_c, STBI_rgb_alpha);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    int global_width = viewport.num_columns * viewport.width;
    int global_height = viewport.num_rows * viewport.height;
    double fov = 45.0;
    double aspect = (double)global_width / (double)global_height;
    double near = 0.1;
    double far = 100.0;
    double frustum_h = tan((fov / 2.0) / 180.0 * M_PI) * near;
    double frustum_w = frustum_h * aspect;
    double horizontal_t1 = (double)viewport.column / (double)viewport.num_columns;
    double horizontal_t2 = (double)(viewport.column + 1) / (double)viewport.num_columns;
    double vertical_t1 = (double)(viewport.num_rows - viewport.row - 1) / (double)viewport.num_rows;
    double vertical_t2 = (double)(viewport.num_rows - viewport.row) / (double)viewport.num_rows;
    double left = (horizontal_t1 * 2.0 * frustum_w) - frustum_w;
    double right = (horizontal_t2 * 2.0 * frustum_w) - frustum_w;
    double bottom = (vertical_t1 * 2.0 * frustum_h) - frustum_h;
    double top = (vertical_t2 * 2.0 * frustum_h) - frustum_h;
    if (app->render_mode == RenderMode::LocalDisplay) // normal render
    {
        app->mat_projection = glm::frustum(left, right, bottom, top, near, far);
    }
    else                                              // flipped render - for capturing image
    {
        app->mat_projection = glm::frustum(left, right, top, bottom, near, far); 
    }

    glUseProgram(shader->program);
    glm::vec3 ambient = glm::vec3(0.2, 0.2, 0.2);
    glm::vec3 diffuse = glm::vec3(1.0, 1.0, 1.0);
    glm::vec3 light_dir = glm::normalize(glm::vec3(0.2, 1.0, 1.0));
    glUniform3fv(shader->ambientcol_uniform, 1, glm::value_ptr(ambient));
    glUniform3fv(shader->lightcol_uniform, 1, glm::value_ptr(diffuse));
    glUniform3fv(shader->lightdir_uniform, 1, glm::value_ptr(light_dir));
    glUseProgram(0);

    app->rotate_x =  30.0;
    app->rotate_y = -45.0;
    if (app->rank == 0)
    {
        app->render_time = MPI_Wtime();
    }
    MPI_Bcast(&(app->render_time), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Idle(GLFWwindow *window, GShaderProgram& shader, AppData& app, LocalViewport& viewport)
{
    Render(window, shader, app, viewport);
}

void Render(GLFWwindow *window, GShaderProgram& shader, AppData& app, LocalViewport& viewport)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    double now;
    if (app.rank == 0)
    {
        now = MPI_Wtime();
    }
    MPI_Bcast(&now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double dt = now - app.render_time;
    app.rotate_x += 10.0 * dt;
    app.rotate_y -= 15.0 * dt;

    app.mat_modelview = glm::translate(glm::mat4(1.0), glm::vec3(0.0, 0.0, -5.0));
    app.mat_modelview = glm::rotate(app.mat_modelview, glm::radians((float)(app.rotate_x)), glm::vec3(1.0, 0.0, 0.0));
    app.mat_modelview = glm::rotate(app.mat_modelview, glm::radians((float)(app.rotate_y)), glm::vec3(0.0, 1.0, 0.0));

    glUseProgram(shader.program);
    SetMatrixUniforms(shader, app);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.tex_id);
    glUniform1i(shader.img_uniform, 0);
    glBindVertexArray(app.vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

    app.render_time = now;

    if (app.render_mode == RenderMode::ImageCapture)
    {
        glReadPixels(0, 0, viewport.width, viewport.height, GL_RGBA, GL_UNSIGNED_BYTE, app.framebuffer);
    }

    app.frame_count++;
    if (app.rank == 0 && app.frame_count % 60 == 0)
    {
        printf("frame time: %.3lf\n", dt);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    glfwSwapBuffers(window);
}

void SetMatrixUniforms(GShaderProgram& shader, AppData& app)
{
    glUniformMatrix4fv(shader.proj_uniform, 1, GL_FALSE, glm::value_ptr(app.mat_projection));
    glUniformMatrix4fv(shader.mv_uniform, 1, GL_FALSE, glm::value_ptr(app.mat_modelview));

    glm::mat3 mat_normal = glm::inverse(app.mat_modelview);
    mat_normal = glm::transpose(mat_normal);

    glUniformMatrix3fv(shader.norm_uniform, 1, GL_FALSE, glm::value_ptr(mat_normal));
}


// Auxillary functions
GLuint CreateCubeVao(AppData& app)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // vertices
    GLuint vertex_position_buffer;
    glGenBuffers(1, &vertex_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);
    GLfloat vertices[72] = {
        // Front face
        -1.0, -1.0,  1.0,
        1.0, -1.0,  1.0,
        1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,

        // Back face
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0, -1.0, -1.0,

        // Top face
        -1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0,
        1.0,  1.0,  1.0,
        1.0,  1.0, -1.0,

        // Bottom face
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, -1.0,  1.0,
        -1.0, -1.0,  1.0,

        // Right face
        1.0, -1.0, -1.0,
        1.0,  1.0, -1.0,
        1.0,  1.0,  1.0,
        1.0, -1.0,  1.0,

        // Left face
        -1.0, -1.0, -1.0,
        -1.0, -1.0,  1.0,
        -1.0,  1.0,  1.0,
        -1.0,  1.0, -1.0
    };
    glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_position_attrib);
    glVertexAttribPointer(app.vertex_position_attrib, 3, GL_FLOAT, false, 0, 0);

    // normals
    GLuint vertex_normal_buffer;
    glGenBuffers(1, &vertex_normal_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer);
    GLfloat normals[72] = {
        // Front
        0.0,  0.0,  1.0,
        0.0,  0.0,  1.0,
        0.0,  0.0,  1.0,
        0.0,  0.0,  1.0,

        // Back
        0.0,  0.0, -1.0,
        0.0,  0.0, -1.0,
        0.0,  0.0, -1.0,
        0.0,  0.0, -1.0,

        // Top
        0.0,  1.0,  0.0,
        0.0,  1.0,  0.0,
        0.0,  1.0,  0.0,
        0.0,  1.0,  0.0,

        // Bottom
        0.0, -1.0,  0.0,
        0.0, -1.0,  0.0,
        0.0, -1.0,  0.0,
        0.0, -1.0,  0.0,

        // Right
        1.0,  0.0,  0.0,
        1.0,  0.0,  0.0,
        1.0,  0.0,  0.0,
        1.0,  0.0,  0.0,

        // Left
        -1.0,  0.0,  0.0,
        -1.0,  0.0,  0.0,
        -1.0,  0.0,  0.0,
        -1.0,  0.0,  0.0
    };
    glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(GLfloat), normals, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_normal_attrib);
    glVertexAttribPointer(app.vertex_normal_attrib, 3, GL_FLOAT, false, 0, 0);

    // textures
    GLuint vertex_texcoord_buffer;
    glGenBuffers(1, &vertex_texcoord_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_texcoord_buffer);
    GLfloat texcoords[48] = {
        // Front
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,

        // Back
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,

        // Top
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,

        // Bottom
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,

        // Right
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0,

        // Left
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0
    };
    glBufferData(GL_ARRAY_BUFFER, 48 * sizeof(GLfloat), texcoords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_texcoord_attrib);
    glVertexAttribPointer(app.vertex_texcoord_attrib, 2, GL_FLOAT, false, 0, 0);

    // faces of the triangles
    GLuint vertex_index_buffer;
    glGenBuffers(1, &vertex_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_index_buffer);
    GLushort indices[36] = {
         0,  1,  2,      0,  2,  3,   // front
         4,  5,  6,      4,  6,  7,   // back
         8,  9, 10,      8, 10, 11,   // top
        12, 13, 14,     12, 14, 15,   // bottom
        16, 17, 18,     16, 18, 19,   // right
        20, 21, 22,     20, 22, 23    // left
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLushort), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);

    return vao;
}

GShaderProgram CreateTextureShader(AppData& app)
{
    GShaderProgram shader;

    const char *vertex_file = "resrc/shaders/texture_phong.vert";
    char *vertex_src;
    int32_t vertex_src_length = ReadFile(vertex_file, &vertex_src);
    GLint vertex_shader = CompileShader(vertex_src, vertex_src_length, GL_VERTEX_SHADER);
    free(vertex_src);

    const char *fragment_file = "resrc/shaders/texture_phong.frag";
    char *fragment_src;
    int32_t fragment_src_length = ReadFile(fragment_file, &fragment_src);
    GLint fragment_shader = CompileShader(fragment_src, fragment_src_length, GL_FRAGMENT_SHADER);
    free(fragment_src);

    CreateShaderProgram(vertex_shader, fragment_shader, &shader.program);

    glBindAttribLocation(shader.program, app.vertex_position_attrib, "aVertexPosition");
    glBindAttribLocation(shader.program, app.vertex_normal_attrib, "aVertexNormal");
    glBindAttribLocation(shader.program, app.vertex_texcoord_attrib, "aVertexTexCoord");
    glBindAttribLocation(shader.program, 0, "FragColor");

    LinkShaderProgram(shader.program);

    shader.mv_uniform = glGetUniformLocation(shader.program, "uModelViewMatrix");
    shader.proj_uniform = glGetUniformLocation(shader.program, "uProjectionMatrix");
    shader.norm_uniform = glGetUniformLocation(shader.program, "uNormalMatrix");
    shader.ambientcol_uniform = glGetUniformLocation(shader.program, "uAmbientColor");
    shader.lightdir_uniform = glGetUniformLocation(shader.program, "uLightingDirection");
    shader.lightcol_uniform = glGetUniformLocation(shader.program, "uDirectionalColor");
    shader.img_uniform = glGetUniformLocation(shader.program, "uImage");

    return shader;
}

GLint CompileShader(char *source, uint32_t length, GLint type)
{
    GLint status;
    GLint shader = glCreateShader(type);

    const char *src_bytes = source;
    const GLint len = length;
    glShaderSource(shader, 1, &src_bytes, &len);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == 0)
    {
        GLint log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        char *info = (char*)malloc(log_length + 1);
        glGetShaderInfoLog(shader, log_length, NULL, info);
        fprintf(stderr, "Error: failed to compile shader:\n%s\n", info);
        free(info);

        return -1;
    }

    return shader;
}

void CreateShaderProgram(GLint vertex_shader, GLint fragment_shader, GLuint *program)
{
    *program = glCreateProgram();
    glAttachShader(*program, vertex_shader);
    glAttachShader(*program, fragment_shader);
}

void LinkShaderProgram(GLuint program)
{
    GLint status;
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == 0)
    {
        fprintf(stderr, "Error: unable to initialize shader program\n");
    }
}

int32_t ReadFile(const char* filename, char** data_ptr)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int32_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *data_ptr = (char*)malloc(fsize);
    size_t read = fread(*data_ptr, fsize, 1, fp);
    if (read != 1)
    {
        fprintf(stderr, "Error cannot read %s\n", filename);
        return -1;
    }

    fclose(fp);

    return fsize;
}

void GetClosestFactors2(int value, int *factor_1, int *factor_2)
{
    int test_num = (int)sqrt(value);
    while (value % test_num != 0)
    {
        test_num--;
    }
    *factor_2 = test_num;
    *factor_1 = value / test_num;
}
