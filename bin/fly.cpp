#include <v4r/display.hpp>
#include <v4r/debug.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <chrono>

using namespace std;
using namespace v4r;

const float mouse_speed = 1e-4;
const float movement_speed = 1;
const float rotate_speed = 0.5;

static GLFWwindow * makeWindow(const glm::u32vec2 &dim)
{
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    return glfwCreateWindow(dim.x, dim.y,
                            "V4R", NULL, NULL);
}

struct CameraState {
    glm::vec3 eye;
    glm::vec3 look;
    glm::vec3 up;
};

static glm::i8vec3 key_movement(0, 0, 0);

void windowKeyHandler(GLFWwindow *window, int key, int, int action, int)
{
    if (action == GLFW_REPEAT) return;

    glm::i8vec3 cur_movement(0, 0, 0);
    switch (key) {
        case GLFW_KEY_ESCAPE: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
            break;
        }
        case GLFW_KEY_ENTER: {
            if (action == GLFW_PRESS) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            }
            break;
        }
        case GLFW_KEY_W: {
            cur_movement.y += 1;
            break;
        }
        case GLFW_KEY_A: {
            cur_movement.x -= 1;
            break;
        }
        case GLFW_KEY_S: {
            cur_movement.y -= 1;
            break;
        }
        case GLFW_KEY_D: {
            cur_movement.x += 1;
            break;
        }
        case GLFW_KEY_Q: {
            cur_movement.z -= 1;
            break;
        }
        case GLFW_KEY_E: {
            cur_movement.z += 1;
            break;
        }
    }

    if (action == GLFW_PRESS) {
        key_movement += cur_movement;
    } else {
        key_movement -= cur_movement;
    }
}

static glm::vec2 cursorPosition(GLFWwindow *window)
{
    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    return glm::vec2(mouse_x, mouse_y);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << argv[0] << " scene" << endl;
        exit(EXIT_FAILURE);
    }

    if (!glfwInit()) {
        cerr << "GLFW failed to initialize" << endl;
        exit(EXIT_FAILURE);
    }

    RenderDoc rdoc;

    RenderFeatures::Outputs outputs =
        RenderFeatures::Outputs::Color;
    RenderFeatures::MeshColor color_src = RenderFeatures::MeshColor::Texture;

    BatchPresentRenderer renderer({0, 1, 1, 1, 1024, 1024,
        glm::mat4(1.f),
        {
            color_src,
            RenderFeatures::Pipeline::Unlit,
            outputs,
            RenderFeatures::Options::DoubleBuffered |
                RenderFeatures::Options::CpuSynchronization
        }
    }, false);

    auto loader = renderer.makeLoader();
    auto scene = loader.loadScene(argv[1]);

    glm::u32vec2 frame_dim = renderer.getFrameDimensions();
    GLFWwindow *window = makeWindow(frame_dim);

    auto cmd_stream = renderer.makeCommandStream(window);

    CameraState cam {
        glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 1, 0)
    };
    glm::vec2 mouse_prev = cursorPosition(window);

    vector<Environment> envs;
    envs.emplace_back(cmd_stream.makeEnvironment(scene, 90));

    envs[0].setCameraView(cam.eye, cam.look, cam.up);

    glfwSetKeyCallback(window, windowKeyHandler);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    rdoc.startFrame();
    RenderSync prevsync = cmd_stream.render(envs);
    rdoc.endFrame();

    auto time_prev = chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        auto time_cur = chrono::steady_clock::now();
        chrono::duration<float> elapsed_duration = time_cur - time_prev;
        time_prev = time_cur;
        float time_delta = elapsed_duration.count();

        glfwPollEvents();
        glm::vec2 mouse_delta; 
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            glm::vec2 mouse_cur = cursorPosition(window);
            mouse_delta = mouse_cur - mouse_prev;
            mouse_prev = mouse_cur;
        } else {
            mouse_delta = glm::vec2(0, 0);
            mouse_prev = cursorPosition(window);
        }

        glm::vec3 to_look = cam.look - cam.eye;
        glm::vec3 right = glm::cross(to_look, cam.up);
        glm::mat3 around_right(glm::angleAxis(-mouse_delta.y * mouse_speed,
                                              right));

        cam.up = around_right * cam.up;

        glm::mat3 around_up(glm::angleAxis(-mouse_delta.x * mouse_speed,
                                           cam.up));

        to_look = around_up * around_right * to_look;

        glm::mat3 around_look(glm::angleAxis(float(key_movement.z) * rotate_speed * time_delta,
                                             to_look));
        cam.up = around_look * cam.up;
        right = around_look * around_up * right;

        glm::vec2 movement = movement_speed * time_delta * glm::vec2(key_movement.x,
                                                                     key_movement.y);
        cam.eye += right * movement.x + to_look * movement.y;

        cam.look = cam.eye + to_look;

        envs[0].setCameraView(cam.eye, cam.look, cam.up);

        rdoc.startFrame();
        RenderSync newsync = cmd_stream.render(envs);
        rdoc.endFrame();
        prevsync.cpuWait();
        prevsync = move(newsync);
    }
}
