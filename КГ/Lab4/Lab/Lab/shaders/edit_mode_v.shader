#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texCoords;
layout(location = 3) in vec3 a_tangent;
layout(location = 4) in vec3 a_bitangent;

uniform mat4 u_mvp;

void main()
{
    gl_Position = u_mvp * vec4(a_position, 1.0f);
}