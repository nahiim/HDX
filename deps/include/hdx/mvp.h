#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct MVP
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec4 view_pos;
};