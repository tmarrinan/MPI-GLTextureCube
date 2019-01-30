#version 150

uniform sampler2D uImage;

in vec2 vTexCoord;
in vec3 vLightWeighting;

out vec4 FragColor;

void main() {
    vec4 textureColor = texture(uImage, vTexCoord);

    FragColor = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
}
