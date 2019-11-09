#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 translate;
// layout(location = 2) in vec3 normal;
// vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
// layout(location = 3) in vec4 model[4]; // per-instance.

layout(set = 0, binding = 0) uniform Args {
    mat4 proj;
    mat4 view;
    mat4 model;
};

layout(location = 0) out vec4 frag_pos;
// layout(location = 1) out vec3 frag_norm;
layout(location = 1) out vec4 frag_color;

void main() {
    // mat4 model_mat = mat4(model[0], model[1], model[2], model[3]);
    frag_color = color;
    // frag_norm = normalize((vec4(normal, 1.0)).xyz);
    mat4 trans_mat = mat4(1.0);
    trans_mat[3] = vec3(translate, 1.0);
    model = model * trans_mat;
    frag_pos = vec4(position, 1.0);
    gl_Position = proj * view * model * frag_pos;
}
