#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>


struct Ray
{
	alignas(16) glm::vec3 origin;
	alignas(16) glm::vec3 direction;
};


//class ray
//{
//public:
//    ray() {}
//
//    ray(const glm::vec3& origin, const glm::vec3& direction) : m_origin(origin), m_direction(direction) {}
//
//    const glm::vec3& origin() const
//    {
//        return m_origin;
//    }
//    const glm::vec3& direction() const
//    {
//        return m_direction;
//    }
//
//    glm::vec3 at(double t) const
//    {
//        return m_origin + static_cast<float>(t) * m_direction;
//    }
//
//private:
//    glm::vec3 m_origin;
//    glm::vec3 m_direction;
//};

#endif