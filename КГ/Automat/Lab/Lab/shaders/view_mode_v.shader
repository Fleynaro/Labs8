#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texCoords;
layout(location = 3) in vec3 a_tangent;
layout(location = 4) in vec3 a_bitangent;

uniform mat3 u_normal_mat;
uniform mat4 u_mv_mat;
uniform mat4 u_projection_mat;

out vec3 v_pos;
out vec3 v_normal;

void main()
{
    v_normal = u_normal_mat * a_normal;
    v_pos = vec3(u_mv_mat * vec4(a_position, 1.0f));
    gl_Position = u_projection_mat * u_mv_mat * vec4(a_position, 1.0f);
}