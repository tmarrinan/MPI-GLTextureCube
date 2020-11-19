#version 150

uniform sampler2D uImage;

in vec2 vTexCoord;

out vec4 FragColor;

void main() {
    vec4 textureColor = texture(uImage, vTexCoord);

    FragColor = vec4(textureColor);
}
