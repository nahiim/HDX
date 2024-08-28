#pragma once

#include <iostream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

class PerspectiveCamera
{
private:
    glm::mat4 projection_matrix;
    glm::mat4 view_matrix;
    glm::vec3 position;

    glm::vec3 world_up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cam_front;
    glm::vec3 cam_up;
    glm::vec3 cam_right;

public:
    PerspectiveCamera()
    {}

    PerspectiveCamera(const glm::vec3& pos, const glm::vec3& front, float fov, float aspect_ratio, float near_plane, float far_plane)
    {
        position = pos;
        cam_front = front;
        view_matrix = glm::lookAt(pos, front, world_up);
        projection_matrix = glm::perspective(fov, aspect_ratio, near_plane, far_plane);
    }

    void translate(const float& dx, const float& dy, const float& dz)
    {
        // Calculate the forward, right, and up vectors
        glm::vec3 forward = glm::normalize(cam_front - position);
        glm::vec3 right = glm::normalize(glm::cross(forward, world_up));
        glm::vec3 cameraUp = glm::normalize(glm::cross(right, forward));

        // Move the camera along the local axes
        position += right * dx;
        position += cameraUp * dy;
        position += forward * dz;

        // Update the camera target based on the new position
        cam_front = position + forward;
        view_matrix = glm::lookAt(position, cam_front, world_up);
    }

    void rotate(float delta_pitch, float delta_yaw, float delta_roll)
    {
        // Convert angles from degrees to radians
        float radYaw = glm::radians(delta_yaw);
        float radPitch = glm::radians(delta_pitch);

        // Calculate the direction vector from the camera position to the target
        glm::vec3 direction = glm::normalize(cam_front - position);

        // Apply yaw rotation around the Y axis
        glm::quat quaternionYaw = glm::angleAxis(radYaw, glm::vec3(0.0f, 1.0f, 0.0f));
        direction = quaternionYaw * direction;

        // Apply pitch rotation around the X axis (use the camera's right vector as the axis)
        glm::vec3 right = glm::normalize(glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::quat quaternionPitch = glm::angleAxis(radPitch, right);
        direction = quaternionPitch * direction;

        // Update the camera target based on the new direction
        cam_front = position + direction;
        view_matrix = glm::lookAt(position, cam_front, world_up);
/*
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
*/
    }

    glm::mat4 getViewMatrix() const
    {
        return view_matrix;
    }
    glm::mat4 getProjectionMatrix() const
    {
        return projection_matrix;
    }
    glm::vec3 getPosition() const
    {
        return position;
    }
};