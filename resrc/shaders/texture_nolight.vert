#version 150

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

in vec3 aVertexPosition;
in vec2 aVertexTexCoord;

out vec2 vTexCoord;

void main() {
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);
    vTexCoord = aVertexTexCoord;
}
