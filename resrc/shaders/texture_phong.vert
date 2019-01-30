#version 150

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
uniform vec3 uAmbientColor;
uniform vec3 uLightingDirection;
uniform vec3 uDirectionalColor;

in vec3 aVertexPosition;
in vec3 aVertexNormal;
in vec2 aVertexTexCoord;

out vec2 vTexCoord;
out vec3 vLightWeighting;

void main() {
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);
    vTexCoord = aVertexTexCoord;

    vec3 transformedNormal = uNormalMatrix * aVertexNormal;
    float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
    vLightWeighting = uAmbientColor + uDirectionalColor * directionalLightWeighting;
}
