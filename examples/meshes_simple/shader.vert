#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 translate;
layout(location = 2) in vec3 color;
layout(location = 3) in uint dir;

// layout(location = 2) in vec3 normal;
// vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
// layout(location = 3) in vec4 model[4]; // per-instance.

layout(set = 0, binding = 0) uniform Args {
    mat4 proj;
    mat4 view;
    mat4 model[6];
};

layout(location = 0) out vec4 frag_pos;
// layout(location = 1) out vec3 frag_norm;
layout(location = 1) out vec4 frag_color;

void main() {
    // mat4 model_mat = mat4(model[0], model[1], model[2], model[3]);
    frag_color = vec4(color, 1.0);
    // frag_norm = normalize((vec4(normal, 1.0)).xyz);
    mat4 trans_mat = mat4(1.0);
    trans_mat[3] = translate; //vec4(translate, 1.0);
    mat4 model2 = trans_mat * model[dir];
    frag_pos = vec4(position, 1.0);
    gl_Position = proj * view * model2 * frag_pos;
}
