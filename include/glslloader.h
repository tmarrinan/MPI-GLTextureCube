#include <iostream>
#include <string>
#include <map>
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace glsl {
    GLuint createShaderProgram(const char *vert_filename, const char *frag_filename);
    void linkShaderProgram(GLuint program);
    void getShaderProgramUniforms(GLuint program, std::map<std::string,GLint>& uniforms);

    static GLint compileShader(char *source, int32_t length, GLenum type);
    static GLuint attachShaders(GLuint shaders[], uint16_t num_shaders);
    static std::string shaderTypeToString(GLenum type);
    static int32_t readFile(const char* filename, char** data_ptr);
}
