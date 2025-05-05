#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

class RayCamera
{
public:
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 right;
    glm::vec3 up;

    RayCamera() = default;

    RayCamera(glm::vec3 pos, glm::vec3 dir)
        : position(pos), direction(glm::normalize(dir))
    {
        right = glm::normalize(glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, direction));
    }

    void translate(const float& dx, const float& dy, const float& dz)
    {
        position += dx * right + dy * up + dz * direction;
    }
    void rotate(float delta_pitch, float delta_yaw, float delta_roll)
    {
        direction = glm::rotate(direction, glm::radians(delta_yaw), up);
        direction = glm::rotate(direction, glm::radians(delta_pitch), right);
        direction = glm::rotate(direction, glm::radians(delta_roll), direction);
        updateVectors();
    }

private:
    void updateVectors()
    {
        right = glm::normalize(glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, direction));
    }
};
