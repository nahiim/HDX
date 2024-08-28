
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <hdx/vk_vertex.h>
#include <hdx/vk_draw_call_info.h>
#include <hdx/mvp.h>


namespace hdx
{
    void mapMemory(vk::Device device, vk::DeviceMemory buffer_memory, uint64_t size, void* in_data);
    vk::Buffer createBuffer(vk::Device device, vk::BufferUsageFlags usage, uint64_t size);
    void allocateBufferMemory(const vk::Device& device, vk::PhysicalDeviceMemoryProperties mem_properties, const vk::Buffer& buffer, vk::DeviceMemory& buffer_memory, vk::MemoryPropertyFlags properties);
    void copyBuffer(vk::Device device, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceMemory src_buffer_memory, uint64_t size, vk::CommandPool cmd_pool, vk::Queue queue);
    vk::Image createTextureImage(vk::Device device, const char* path, uint32_t mip_levels, vk::SampleCountFlagBits msaa_samples, vk::DeviceMemory& textureImageMemory, DeviceDesc device_desc, vk::CommandPool cmd_pool, vk::Queue queue);
    vk::ImageView createTextureImageView(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels);
    vk::Sampler createTextureSampler(vk::Device device, vk::PhysicalDeviceProperties properties, uint32_t mip_levels);
    void updateDescriptorSet(vk::Device device, vk::DescriptorSet& descriptorSet, vk::Buffer& uniformBuffer, uint64_t size, vk::ImageView textureImageView, vk::Sampler textureSampler);
    vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device);
    vk::DescriptorPool createImageDescriptorPool(vk::Device device, uint32_t count);
    vk::DescriptorSet allocateDescriptorSet(vk::Device device, vk::DescriptorSetLayout descriptorSetLayout, vk::DescriptorPool descriptorPool);
    vk::Pipeline createGraphicsPipeline(const vk::Device& device, vk::PipelineLayout&, const vk::RenderPass&, vk::SampleCountFlagBits, const std::string&, const std::string&, const vk::VertexInputBindingDescription&, const std::vector <vk::VertexInputAttributeDescription>&, vk::DescriptorSetLayout, vk::PrimitiveTopology);
    void destroyAndFree(vk::Device&, vk::Buffer&, vk::DeviceMemory&);
}

using namespace hdx;

class Sprite
{
public:
    // Default Constructor
    Sprite(){}


    void init(vk::Device &l_device, const char* path, uint32_t mip_levels, vk::SampleCountFlagBits msaa_samples, vk::RenderPass renderpass, uint32_t swapchain_size,
        DeviceDesc device_desc,
        vk::CommandPool &cmd_pool, vk::Queue &queue, VPMatrices vp, uint64_t uniform_buffer_size)
    {
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        dci.dset_layout = createDescriptorSetLayout(l_device);
        dci.pipeline = createGraphicsPipeline(l_device, dci.pipelineLayout, renderpass, msaa_samples, "res/shaders/vertex_shader.spv", "res/shaders/fragment_shader.spv", bindingDescription, attributeDescriptions, dci.dset_layout, vk::PrimitiveTopology::eTriangleList);

        stg_vertex_buffer = createBuffer(l_device, vk::BufferUsageFlagBits::eTransferSrc, sizeof_vertices);
        allocateBufferMemory(l_device, device_desc.memory_properties, stg_vertex_buffer, stg_vertex_buffer_memory, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        dci.vertexBuffer = createBuffer(l_device, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, sizeof_vertices);
        allocateBufferMemory(l_device, device_desc.memory_properties, dci.vertexBuffer, vertex_buffer_memory, vk::MemoryPropertyFlagBits::eDeviceLocal);
        
        stg_index_buffer = createBuffer(l_device, vk::BufferUsageFlagBits::eTransferSrc, sizeof_indices);
        allocateBufferMemory(l_device, device_desc.memory_properties, stg_index_buffer, stg_index_buffer_memory, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        dci.indexBuffer = createBuffer(l_device, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, sizeof_indices);
        allocateBufferMemory(l_device, device_desc.memory_properties, dci.indexBuffer, index_buffer_memory, vk::MemoryPropertyFlagBits::eDeviceLocal);

        mapMemory(l_device, stg_index_buffer_memory, sizeof_indices, (void*)indices.data());
        copyBuffer(l_device, stg_index_buffer, dci.indexBuffer, stg_index_buffer_memory, sizeof_indices, cmd_pool, queue);
        destroyAndFree(l_device, stg_index_buffer, stg_index_buffer_memory);

        texture_image = createTextureImage(l_device, path, mip_levels, msaa_samples, texture_image_memory, device_desc, cmd_pool, queue);
        texture_image_view = createTextureImageView(l_device, texture_image, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mip_levels);
        texture_sampler = createTextureSampler(l_device, device_desc.properties, mip_levels);

        uniform_buffer = createBuffer(l_device, vk::BufferUsageFlagBits::eUniformBuffer, uniform_buffer_size);
        allocateBufferMemory(l_device, device_desc.memory_properties, uniform_buffer, uniform_buffer_memory, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        mapMemory(l_device, uniform_buffer_memory, uniform_buffer_size, &vp);


        dci.dsc_pool = createImageDescriptorPool(l_device, swapchain_size);

            dci.dsc_set = allocateDescriptorSet(l_device, dci.dset_layout, dci.dsc_pool);
            updateDescriptorSet(l_device, dci.dsc_set, uniform_buffer, sizeof(vp), texture_image_view, texture_sampler);

        dci.indexCount = indices.size();
    }

    ~Sprite()
    {}

    void update(vk::Device l_device, vk::PhysicalDeviceMemoryProperties device_memory_properties, vk::CommandPool cmd_pool, vk::Queue queue)
    {
        mapMemory(l_device, stg_vertex_buffer_memory, sizeof_vertices, vertices.data());
        copyBuffer(l_device, stg_vertex_buffer, dci.vertexBuffer, stg_vertex_buffer_memory, sizeof_vertices, cmd_pool, queue);
    }

    void rotate( float angle)
    {
        glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(angle), glm::vec3(0.0f, 0.0f, 1.0f));

        for (Vertex& vert : vertices)
        {
            vert.position = glm::vec3(rotationMatrix * glm::vec4(vert.position, 1.0f));
        }
    }


    void cleanUp(vk::Device& l_device)
    {   
        destroyAndFree(l_device, stg_vertex_buffer, stg_vertex_buffer_memory);

        l_device.destroySampler(texture_sampler);
        l_device.destroyImageView(texture_image_view);

        l_device.destroyImage(texture_image);
        l_device.freeMemory(texture_image_memory);

        l_device.destroyBuffer(uniform_buffer);
        l_device.freeMemory(uniform_buffer_memory);

        l_device.destroyBuffer(dci.vertexBuffer);
        l_device.freeMemory(vertex_buffer_memory);

        l_device.destroyBuffer(dci.indexBuffer);
        l_device.freeMemory(index_buffer_memory);

        l_device.destroyDescriptorPool(dci.dsc_pool);
        l_device.destroyDescriptorSetLayout(dci.dset_layout);

        l_device.destroyPipeline(dci.pipeline);
        l_device.destroyPipelineLayout(dci.pipelineLayout);
    }

    DrawCallInfo dci;


    const std::vector<uint32_t> indices =
    {
        0, 1, 2, 2, 3, 0
    };

    vk::ImageView texture_image_view;
    vk::Sampler texture_sampler;

    vk::Buffer uniform_buffer;
    vk::DeviceMemory uniform_buffer_memory;

private:

    std::vector<Vertex> vertices =
    {
//			position				color					texcoord
        { {-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f} },
        { { 0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.5f}, {1.0f, 0.0f} },
        { { 0.5f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f} },
        { {-0.5f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f} }
    };

    vk::DeviceMemory vertex_buffer_memory;
    vk::Buffer stg_vertex_buffer;
    vk::DeviceMemory stg_vertex_buffer_memory;

    vk::DeviceMemory index_buffer_memory;
    vk::Buffer stg_index_buffer;
    vk::DeviceMemory stg_index_buffer_memory;

    vk::Image texture_image;
    vk::DeviceMemory texture_image_memory;

    uint64_t sizeof_vertices = sizeof(vertices[0]) * vertices.size();

    uint64_t sizeof_indices = sizeof(indices[0]) * indices.size();
};