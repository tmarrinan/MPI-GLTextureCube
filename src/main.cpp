#include <iostream>
#include <cmath>
#include <string>
#include <map>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <IceT.h>
#include <IceTMPI.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "glslloader.h"

#define WINDOW_TITLE "Texture Cube (IceT)"

typedef struct AppData {
    int window_width;
    int window_height;
    GLFWwindow *window;
    int rank;
    int num_proc;
    IceTCommunicator comm;
    IceTContext context;
    IceTImage image;
    glm::vec4 background_color;
    glm::dmat4 mat_projection;
    glm::dmat4 mat_modelview;
    GLuint program;
    std::map<std::string,GLint> uniforms;
    GLuint vertex_position_attrib;
    GLuint vertex_normal_attrib;
    GLuint vertex_texcoord_attrib;
    GLuint framebuffer;
    GLuint framebuffer_texture;
    GLuint framebuffer_depth;
    GLuint cube_vertex_array;
    GLuint plane_vertex_array;
    GLuint tex_id;
} AppData;


void init();
void doFrame();
void render(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
    const IceTFloat *background_color, const IceTInt *readback_viewport, IceTImage result);
void display();
void mat4ToFloatArray(glm::dmat4 mat4, float array[16]);
void mat3ToFloatArray(glm::dmat3 mat3, float array[9]);
void loadShader(std::string shader_filename_base);
GLuint cubeVertexArray();
GLuint planeVertexArray();
void writePpm(const char *filename, int width, int height, const uint8_t *pixels);

AppData app;

int main(int argc, char **argv)
{
    // Initialize MPI
    int rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &(app.rank));
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &(app.num_proc));
    if (rc != 0)
    {
        fprintf(stderr, "Error initializing MPI\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read command line parameters for overall width / height
    app.window_width = 1280;
    app.window_height = 720;
    if (argc >= 2) app.window_width = std::stoi(argv[1]);
    if (argc >= 3) app.window_height = std::stoi(argv[2]);

    // Initialize GLFW
    if (!glfwInit())
    {
        exit(1);
    }

    // Create a window and its OpenGL context
    char title[32];
    snprintf(title, 32, "%s (%d)", WINDOW_TITLE, app.rank);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (app.rank == 0)
    {
        app.window = glfwCreateWindow(app.window_width, app.window_height, title, NULL, NULL);
    }
    else
    {
        app.window = glfwCreateWindow(256, 64, title, NULL, NULL);
    }

    // Make window's context current
    glfwMakeContextCurrent(app.window);
    glfwSwapInterval(1);

    // Initialize GLAD OpenGL extension handling
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        exit(1);
    }

    // Initialize app
    init();

    // main render loop
    doFrame();
    while (!glfwWindowShouldClose(app.window))
    {
        glfwPollEvents();
        //Idle(window, shader, app, m_viewport);
    }

    // clean up
    icetDestroyMPICommunicator(app.comm);
    icetDestroyContext(app.context);
    glfwDestroyWindow(app.window);
    glfwTerminate();

    return 0;
}

void init()
{
    // Initialize IceT
    app.comm = icetCreateMPICommunicator(MPI_COMM_WORLD);
    app.context = icetCreateContext(app.comm);

    // Set IceT window configurations
    icetResetTiles();
    icetAddTile(0, 0, app.window_width, app.window_height, 0);

    // Set IceT compositing strategy
    icetStrategy(ICET_STRATEGY_SEQUENTIAL); // best for a single tile
    //icetStrategy(ICET_STRATEGY_REDUCE); // good all around performance for multiple tiles

    // Set IceT framebuffer settings
    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);

    // Set IceT draw callback (main render function) 
    icetDrawCallback(render);

    // Set background color
    app.background_color = glm::vec4(0.9, 0.9, 0.9, 1.0);

    // Create projection and modelview matrices
    app.mat_projection = glm::perspective(M_PI / 4.0, (double)app.window_width / (double)app.window_height, 0.1, 100.0);
    app.mat_modelview = glm::translate(glm::dmat4(1.0), glm::dvec3(0.0, 0.0, -5.0));
    //app.mat_projection = glm::dmat4(1.0);
    //app.mat_modelview = glm::dmat4(1.0);

    // Set OpenGL settings
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);
    glEnable(GL_DEPTH_TEST);

    // Initialize vertex attributes
    app.vertex_position_attrib = 0;
    app.vertex_normal_attrib = 1;
    app.vertex_texcoord_attrib = 2;

    // Load shader program
    loadShader("resrc/shaders/texture_phong");

    // Create vertex array objects
    app.cube_vertex_array = cubeVertexArray();
    app.plane_vertex_array = planeVertexArray();

    // Load cube texture from JPEG file
    glGenTextures(1, &(app.tex_id));
    glBindTexture(GL_TEXTURE_2D, app.tex_id);
    int img_w, img_h, img_c;
    stbi_set_flip_vertically_on_load(true);
    uint8_t *pixels = stbi_load("resrc/images/crate.jpg", &img_w, &img_h, &img_c, STBI_rgb_alpha);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create offscreen framebuffer
    glGenTextures(1, &(app.framebuffer_texture));
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &(app.framebuffer_depth));
    glBindRenderbuffer(GL_RENDERBUFFER, app.framebuffer_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, app.window_width, app.window_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &(app.framebuffer));
    glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, app.framebuffer_texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, app.framebuffer_depth);
    GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, draw_buffers);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Upload static uniforms
    glUseProgram(app.program);
    glm::vec3 ambient = glm::vec3(0.2, 0.2, 0.2);
    glm::vec3 diffuse = glm::vec3(1.0, 1.0, 1.0);
    glm::vec3 light_dir = glm::normalize(glm::vec3(0.2, 1.0, 1.0));
    glUniform3fv(app.uniforms["uAmbientColor"], 1, glm::value_ptr(ambient));
    glUniform3fv(app.uniforms["uDirectionalColor"], 1, glm::value_ptr(diffuse));
    glUniform3fv(app.uniforms["uLightingDirection"], 1, glm::value_ptr(light_dir));
    glUseProgram(0);
}

void doFrame()
{
    // Animation here

    // Offscreen render and composit
    app.image = icetDrawFrame(glm::value_ptr(app.mat_projection),
                              glm::value_ptr(app.mat_modelview),
                              glm::value_ptr(app.background_color));

    // Render composited image to fullscreen quad on screen of rank 0
    display();
}

void render(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
    const IceTFloat *background_color, const IceTInt *readback_viewport, IceTImage result)
{
    printf("rendering %d [%dx%d]\n", app.rank, icetImageGetWidth(result), icetImageGetHeight(result));

    //glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(app.program);

    
    glm::dmat3 mat_normal = glm::inverse(app.mat_modelview);
    mat_normal = glm::transpose(mat_normal);
    float mat4_proj[16];
    float mat4_mv[16];
    float mat3_norm[9];
    mat4ToFloatArray(app.mat_projection, mat4_proj);
    mat4ToFloatArray(app.mat_modelview, mat4_mv);
    mat3ToFloatArray(mat_normal, mat3_norm);
    glUniformMatrix4fv(app.uniforms["uProjectionMatrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.uniforms["uModelViewMatrix"], 1, GL_FALSE, mat4_mv);
    glUniformMatrix3fv(app.uniforms["uNormalMatrix"], 1, GL_FALSE, mat3_norm);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.tex_id);
    glUniform1i(app.uniforms["uImage"], 0);

    glBindVertexArray(app.cube_vertex_array);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);



    IceTUByte *pixels = icetImageGetColorub(result);
    IceTFloat *depth = icetImageGetDepthf(result);
    uint8_t bg[] = {(uint8_t)(background_color[0] * 255.0f), (uint8_t)(background_color[1] * 255.0f),
                    (uint8_t)(background_color[2] * 255.0f), (uint8_t)(background_color[3] * 255.0f)};
    for (int i = 0; i < app.window_width * app.window_height; i++)
    {
        memcpy(pixels + (i * 4), bg, 4);
        depth[i] = 1.0;
    }
    switch (app.rank) {
        case 0:
        {
            uint8_t red[] {250, 10, 10, 255, 250, 10, 10, 255};
            memcpy(pixels, red, 8);
            memcpy(pixels + (app.window_width * 4), red, 8);
            depth[0] = 0.0;
            depth[1] = 0.0;
            depth[app.window_width] = 0.0;
            depth[app.window_width + 1] = 0.0;
            break;
        }
        case 1:
        {
            uint8_t green[] {10, 200, 40, 255, 10, 200, 40, 255};
            memcpy(pixels + (app.window_width * 2), green, 8);
            memcpy(pixels + (app.window_width * 6), green, 8);
            depth[app.window_width / 2] = 0.0;
            depth[app.window_width / 2 + 1] = 0.0;
            depth[app.window_width * 3 / 2] = 0.0;
            depth[app.window_width * 3 / 2 + 1] = 0.0;
            break;
        }
        case 2:
        {
            uint8_t blue[] {10, 40, 220, 255, 10, 40, 220, 255};
            memcpy(pixels + (app.window_width * 16), blue, 8);
            memcpy(pixels + (app.window_width * 20), blue, 8);
            depth[app.window_width * 4] = 0.0;
            depth[app.window_width * 4 + 1] = 0.0;
            depth[app.window_width * 5] = 0.0;
            depth[app.window_width * 5 + 1] = 0.0;
            break;
        }
        case 3:
        {
            uint8_t black[] {10, 10, 10, 255, 10, 10, 10, 255};
            memcpy(pixels + (app.window_width * 18), black, 8);
            memcpy(pixels + (app.window_width * 22), black, 8);
            depth[app.window_width * 9 / 2] = 0.0;
            depth[app.window_width * 9 / 2 + 1] = 0.0;
            depth[app.window_width * 11 / 2] = 0.0;
            depth[app.window_width * 11 / 2 + 1] = 0.0;
            break;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void display()
{
    /*
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.rank == 0)
    {
        IceTUByte *pixels = icetImageGetColorub(app.image);
        // TODO:
        //  1. upload pixels to texture  -  glTexSubImage2D()
        //  2. draw fullscreen quad w/ texture  -  glDrawElements()

        writePpm("frame.ppm", app.window_width, app.window_height, pixels);
    }
    */

    // Synchronize and display
    MPI_Barrier(MPI_COMM_WORLD);
    glfwSwapBuffers(app.window);
}

void mat4ToFloatArray(glm::dmat4 mat4, float array[16])
{
    array[0] = mat4[0][0];
    array[1] = mat4[0][1];
    array[2] = mat4[0][2];
    array[3] = mat4[0][3];
    array[4] = mat4[1][0];
    array[5] = mat4[1][1];
    array[6] = mat4[1][2];
    array[7] = mat4[1][3];
    array[8] = mat4[2][0];
    array[9] = mat4[2][1];
    array[10] = mat4[2][2];
    array[11] = mat4[2][3];
    array[12] = mat4[3][0];
    array[13] = mat4[3][1];
    array[14] = mat4[3][2];
    array[15] = mat4[3][3];
}

void mat3ToFloatArray(glm::dmat3 mat3, float array[9])
{
    array[0] = mat3[0][0];
    array[1] = mat3[0][1];
    array[2] = mat3[0][2];
    array[3] = mat3[1][0];
    array[4] = mat3[1][1];
    array[5] = mat3[1][2];
    array[6] = mat3[2][0];
    array[7] = mat3[2][1];
    array[8] = mat3[2][2];
}

void loadShader(std::string shader_filename_base)
{
    // Create shader program
    std::string vert_filename = shader_filename_base + ".vert";
    std::string frag_filename = shader_filename_base + ".frag";
    app.program = glsl::createShaderProgram(vert_filename.c_str(), frag_filename.c_str());

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(app.program, app.vertex_position_attrib, "aVertexPosition");
    glBindAttribLocation(app.program, app.vertex_normal_attrib, "aVertexNormal");
    glBindAttribLocation(app.program, app.vertex_texcoord_attrib, "aVertexTexCoord");
    glBindFragDataLocation(app.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(app.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(app.program, app.uniforms);
}

GLuint cubeVertexArray()
{
    // Create vertex array object
    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    // Vertex positions
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

    // Vertex normals
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

    // Vertex texture coordinates
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

    // Faces of the triangles
    GLuint vertex_index_buffer;
    glGenBuffers(1, &vertex_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_index_buffer);
    GLushort indices[36] = {
         0,  1,  2,    0,  2,  3,  // front
         4,  5,  6,    4,  6,  7,  // back
         8,  9, 10,    8, 10, 11,  // top
        12, 13, 14,   12, 14, 15,  // bottom
        16, 17, 18,   16, 18, 19,  // right
        20, 21, 22,   20, 22, 23   // left
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLushort), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);

    return vertex_array;
}

GLuint planeVertexArray()
{
    // Create vertex array object
    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    // Vertex positions
    GLuint vertex_position_buffer;
    glGenBuffers(1, &vertex_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer);
    GLfloat vertices[12] = {
        -1.0, -1.0,  0.0,
         1.0, -1.0,  0.0,
         1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0
    };
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_position_attrib);
    glVertexAttribPointer(app.vertex_position_attrib, 3, GL_FLOAT, false, 0, 0);

    // Vertex texture coordinates
    GLuint vertex_texcoord_buffer;
    glGenBuffers(1, &vertex_texcoord_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_texcoord_buffer);
    GLfloat texcoords[8] = {
        0.0,  0.0,
        1.0,  0.0,
        1.0,  1.0,
        0.0,  1.0
    };
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texcoords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(app.vertex_texcoord_attrib);
    glVertexAttribPointer(app.vertex_texcoord_attrib, 2, GL_FLOAT, false, 0, 0);

    // Faces of the triangles
    GLuint vertex_index_buffer;
    glGenBuffers(1, &vertex_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_index_buffer);
    GLushort indices[6] = {
         0,  1,  2,    0,  2,  3
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLushort), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);

    return vertex_array;
}

void writePpm(const char *filename, int width, int height, const uint8_t *pixels)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: could not create file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    int i;
    for (i = 0; i < width * height; i++)
    {
        fprintf(fp, "%c%c%c", pixels[i * 4 + 0], pixels[i * 4 + 1], pixels[i * 4 + 2]);
    }

    fclose(fp);
}

/*
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
*/
