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

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define WINDOW_TITLE "Texture Cube (IceT)"

typedef struct GlslProgram {
    GLuint program;
    std::map<std::string,GLint> uniforms;
} GlslProgram;

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
    double rotate_x;
    double rotate_y;
    double render_time;
    glm::dvec3 box_position;
    GlslProgram phong;
    GlslProgram nolight;
    GLuint vertex_position_attrib;
    GLuint vertex_normal_attrib;
    GLuint vertex_texcoord_attrib;
    GLuint framebuffer;
    GLuint framebuffer_texture;
    GLuint framebuffer_depth;
    GLuint cube_vertex_array;
    GLuint plane_vertex_array;
    GLuint box_texture;
    GLuint composite_texture;
} AppData;


void init();
void doFrame();
void render(const IceTDouble *projection_matrix, const IceTDouble *modelview_matrix,
    const IceTFloat *background_color, const IceTInt *readback_viewport, IceTImage result);
void display();
void mat4ToFloatArray(glm::dmat4 mat4, float array[16]);
void mat3ToFloatArray(glm::dmat3 mat3, float array[9]);
void loadPhongShader(std::string shader_filename_base);
void loadNoLightShader(std::string shader_filename_base);
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
        app.window = glfwCreateWindow(320, 180, title, NULL, NULL);
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
    uint16_t should_close = 0;
    while (!should_close)
    {
        // poll for user events
        glfwPollEvents();

        // check if any window has been closed
        uint16_t close_this = glfwWindowShouldClose(app.window);
        MPI_Allreduce(&close_this, &should_close, 1, MPI_UINT16_T, MPI_SUM, MPI_COMM_WORLD);

        // render next frame
        doFrame();
    }

    // clean up
    icetDestroyMPICommunicator(app.comm);
    icetDestroyContext(app.context);
    glfwDestroyWindow(app.window);
    glfwTerminate();
    
    MPI_Finalize();

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
    switch (app.rank) {
        case 0:
            app.box_position = glm::dvec3(-1.5, -1.5, -8.0);
            break;
        case 1:
            app.box_position = glm::dvec3(0.0, -1.5, -11.0);
            break;
        case 2:
            app.box_position = glm::dvec3(-1.5, 1.5, -8.0);
            break;
        case 3:
            app.box_position = glm::dvec3(0.0, 1.5, -11.0);
            break;
    }
    //app.mat_projection = glm::dmat4(1.0);
    //app.mat_modelview = glm::dmat4(1.0);

    // Initialize box rotations and animation time
    app.rotate_x =  30.0;
    app.rotate_y = -45.0;
    if (app.rank == 0)
    {
        app.render_time = MPI_Wtime();
    }
    MPI_Bcast(&(app.render_time), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Set OpenGL settings
    glClearColor(app.background_color[0], app.background_color[1], app.background_color[2], app.background_color[3]);
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, app.window_width, app.window_height);

    // Initialize vertex attributes
    app.vertex_position_attrib = 0;
    app.vertex_normal_attrib = 1;
    app.vertex_texcoord_attrib = 2;

    // Load shader programs
    loadPhongShader("resrc/shaders/texture_phong");
    loadNoLightShader("resrc/shaders/texture_nolight");

    // Create vertex array objects
    app.cube_vertex_array = cubeVertexArray();
    app.plane_vertex_array = planeVertexArray();

    // Load cube texture from JPEG file
    glGenTextures(1, &(app.box_texture));
    glBindTexture(GL_TEXTURE_2D, app.box_texture);
    int img_w, img_h, img_c;
    stbi_set_flip_vertically_on_load(true);
    char imgname[32];
    snprintf(imgname, 32, "resrc/images/crate%d.jpg", app.rank);
    uint8_t *pixels = stbi_load(imgname, &img_w, &img_h, &img_c, STBI_rgb_alpha);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create composite texture (for display of final image)
    glGenTextures(1, &(app.composite_texture));
    glBindTexture(GL_TEXTURE_2D, app.composite_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Create offscreen framebuffer
    glGenTextures(1, &(app.framebuffer_texture));
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, app.window_width, app.window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &(app.framebuffer_depth));
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, app.window_width, app.window_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &(app.framebuffer));
    glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, app.framebuffer_texture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, app.framebuffer_depth, 0);
    GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, draw_buffers);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Upload static uniforms
    glUseProgram(app.phong.program);
    glm::vec3 ambient = glm::vec3(0.2, 0.2, 0.2);
    glm::vec3 diffuse = glm::vec3(1.0, 1.0, 1.0);
    glm::vec3 light_dir = glm::normalize(glm::vec3(0.2, 1.0, 1.0));
    glUniform3fv(app.phong.uniforms["uAmbientColor"], 1, glm::value_ptr(ambient));
    glUniform3fv(app.phong.uniforms["uDirectionalColor"], 1, glm::value_ptr(diffuse));
    glUniform3fv(app.phong.uniforms["uLightingDirection"], 1, glm::value_ptr(light_dir));
    
    glUseProgram(app.nolight.program);
    glm::mat4 identity = glm::mat4(1.0);
    glUniformMatrix4fv(app.nolight.uniforms["uProjectionMatrix"], 1, GL_FALSE, glm::value_ptr(identity));
    glUniformMatrix4fv(app.nolight.uniforms["uModelViewMatrix"], 1, GL_FALSE, glm::value_ptr(identity));
    
    glUseProgram(0);
}

void doFrame()
{
    // Animation
    double now;
    if (app.rank == 0)
    {
        now = MPI_Wtime();
    }
    MPI_Bcast(&now, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double dt = now - app.render_time;
    app.rotate_x += 10.0 * dt;
    app.rotate_y -= 15.0 * dt;

    app.mat_modelview = glm::translate(glm::dmat4(1.0), app.box_position);
    app.mat_modelview = glm::rotate(app.mat_modelview, glm::radians(app.rotate_x), glm::dvec3(1.0, 0.0, 0.0));
    app.mat_modelview = glm::rotate(app.mat_modelview, glm::radians(app.rotate_y), glm::dvec3(0.0, 1.0, 0.0));

    app.render_time = now;

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
    // Get render dimensions
    int image_width = icetImageGetWidth(result);
    int image_height = icetImageGetHeight(result);
    //printf("[rank %d] rendering %dx%d\n", app.rank, image_width, image_height);

    // Render
    glBindFramebuffer(GL_FRAMEBUFFER, app.framebuffer);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(app.phong.program);

    glm::dmat3 mat_normal = glm::inverse(app.mat_modelview);
    mat_normal = glm::transpose(mat_normal);
    float mat4_proj[16];
    float mat4_mv[16];
    float mat3_norm[9];
    mat4ToFloatArray(app.mat_projection, mat4_proj);
    mat4ToFloatArray(app.mat_modelview, mat4_mv);
    mat3ToFloatArray(mat_normal, mat3_norm);
    glUniformMatrix4fv(app.phong.uniforms["uProjectionMatrix"], 1, GL_FALSE, mat4_proj);
    glUniformMatrix4fv(app.phong.uniforms["uModelViewMatrix"], 1, GL_FALSE, mat4_mv);
    glUniformMatrix3fv(app.phong.uniforms["uNormalMatrix"], 1, GL_FALSE, mat3_norm);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, app.box_texture);
    glUniform1i(app.phong.uniforms["uImage"], 0);

    glBindVertexArray(app.cube_vertex_array);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Copy image to IceT buffer
    IceTUByte *pixels = icetImageGetColorub(result);
    IceTFloat *depth = icetImageGetDepthf(result);

    glBindTexture(GL_TEXTURE_2D, app.framebuffer_texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, app.framebuffer_depth);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    glBindTexture(GL_TEXTURE_2D, 0);

    //char outname[32];
    //snprintf(outname, 32, "frame_%d.ppm", app.rank);
    //writePpm(outname, image_width, image_height, pixels);
}

void display()
{
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (app.rank == 0)
    {
        glUseProgram(app.nolight.program);
   
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, app.composite_texture);
        IceTUByte *pixels = icetImageGetColorub(app.image);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.window_width, app.window_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glUniform1i(app.nolight.uniforms["uImage"], 0);

        glBindVertexArray(app.plane_vertex_array);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);

        //writePpm("frame.ppm", app.window_width, app.window_height, pixels);

        glUseProgram(0);
    }
    
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

void loadPhongShader(std::string shader_filename_base)
{
    // Create shader program
    std::string vert_filename = shader_filename_base + ".vert";
    std::string frag_filename = shader_filename_base + ".frag";
    app.phong.program = glsl::createShaderProgram(vert_filename.c_str(), frag_filename.c_str());

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(app.phong.program, app.vertex_position_attrib, "aVertexPosition");
    glBindAttribLocation(app.phong.program, app.vertex_normal_attrib, "aVertexNormal");
    glBindAttribLocation(app.phong.program, app.vertex_texcoord_attrib, "aVertexTexCoord");
    glBindFragDataLocation(app.phong.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(app.phong.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(app.phong.program, app.phong.uniforms);
}

void loadNoLightShader(std::string shader_filename_base)
{
    // Create shader program
    std::string vert_filename = shader_filename_base + ".vert";
    std::string frag_filename = shader_filename_base + ".frag";
    app.nolight.program = glsl::createShaderProgram(vert_filename.c_str(), frag_filename.c_str());

    // Specify input and output attributes for the GPU program
    glBindAttribLocation(app.nolight.program, app.vertex_position_attrib, "aVertexPosition");
    glBindAttribLocation(app.nolight.program, app.vertex_texcoord_attrib, "aVertexTexCoord");
    glBindFragDataLocation(app.nolight.program, 0, "FragColor");

    // Link compiled GPU program
    glsl::linkShaderProgram(app.nolight.program);

    // Get handles to uniform variables defined in the shaders
    glsl::getShaderProgramUniforms(app.nolight.program, app.nolight.uniforms);
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
