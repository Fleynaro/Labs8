#version 330 core
layout(location = 0) out vec4 o_color;

in vec3 v_pos;
in vec3 v_normal;

uniform vec3 u_cam_pos;
uniform vec3 u_color;

void main()
{
    float S = 30;
    vec3 color = u_color;
    vec3 n = normalize(v_normal);
    vec3 E = u_cam_pos;
    vec3 L = u_cam_pos;
    vec3 l = normalize(v_pos - L);
    float d = max(dot(n, -l), 0.3);
    vec3 e = normalize(E - v_pos);
    vec3 h = normalize(-l + e);
    float s = pow(max(dot(n, h), 0.0), S) * 0.5;
    vec3 final_color = color * d + s * vec3(1, 1, 1);
    o_color = vec4(pow(final_color, vec3(1.0 / 2.2)), 1);
}