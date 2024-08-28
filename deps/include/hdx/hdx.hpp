
#pragma once

#include <iostream>
#include <cstring>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <optional>
#include <map>
#include <set>
#include <vector>
#include <fstream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include<vulkan/vulkan.hpp>

#include "stb_image.h"
#include "window.h"

namespace hdx
{
	struct DeviceDesc
	{
		vk::PhysicalDevice physical_device;
		std::vector<vk::QueueFamilyProperties> queue_families;
		vk::PhysicalDeviceMemoryProperties memory_properties;
		vk::PhysicalDeviceProperties properties;
		vk::PhysicalDeviceFeatures features;
		vk::FormatProperties format_properties;
	};
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphics_family;
		std::optional<uint32_t> present_family;
		std::optional<uint32_t> transfer_family;
		std::optional<uint32_t> compute_family;

		/*
			bool is_complete()
			{
				return graphics_family.has_value() && present_family.has_value();
			}
		*/
	};
	struct BufferDesc
	{
		vk::Buffer buffer;
		vk::DeviceMemory memory;
		uint64_t size;
	};
	struct ImageDesc
	{
		vk::Image image;
		vk::ImageView imageview;
		vk::DeviceMemory memory;
	};


    vk::VertexInputBindingDescription getBindingDescription(uint32_t binding, uint32_t stride, vk::VertexInputRate input_rate);
    vk::VertexInputAttributeDescription getAttributeDescription(uint32_t binding, uint32_t location, vk::Format format, uint32_t offset);

	std::vector<char> read_file(const std::string& filename);

	bool checkValidationLayerSupport(const std::vector<const char*> validation_layers);
	void createInstance(vk::Instance &instance, Window *window, const char* app_name, bool enable_validation_layers, const std::vector<const char*> validation_layers);

	VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
	void createDebugMessenger(vk::DebugUtilsMessengerEXT& debug_messenger, vk::Instance instance, vk::DispatchLoaderDynamic dldi);



// MEMORY FUNCTIONS
    uint32_t findMemoryType(vk::PhysicalDeviceMemoryProperties& mem_properties, uint32_t type_filter, vk::MemoryPropertyFlags properties);
    void copyToDevice(vk::Device device, BufferDesc buffer, void* source, uint64_t size);

// PIPELINE FUNCTIONS
	vk::ShaderModule createShaderModule(const vk::Device& device, const std::vector<char>& code);
	vk::Pipeline createGraphicsPipeline(const vk::Device& device, vk::PipelineLayout& pipeline_layout, const vk::RenderPass& rp, vk::SampleCountFlagBits msaa_samples, const std::string& vertex_shader, const std::string& fragment_shader, const std::vector<vk::VertexInputBindingDescription>& binding_descriptions, const std::vector<vk::VertexInputAttributeDescription>& attribute_descriptions, vk::DescriptorSetLayout dset_layout, vk::PrimitiveTopology topology, vk::Extent2D extent);
	vk::Pipeline createComputePipeline(const vk::Device& device, vk::DescriptorSetLayout dset_layout, vk::PipelineLayout& pipeline_layout, const std::string& path);

// RENDERPASS FUNCTIONS
	vk::RenderPass createRenderpass(vk::Device device, vk::SampleCountFlagBits msaa_samples, vk::Format format);
	void beginRenderpass(vk::CommandBuffer cmd_buffer, vk::RenderPass& renderpass, vk::Framebuffer framebuffer, uint32_t width, uint32_t height, std::vector<vk::ClearValue> clear_values);
	vk::RenderPass createDepthRenderpass(vk::Device device);

// SWAPCHAIN FUNCTIONS
    vk::SwapchainKHR createSwapchain(vk::Device device, vk::SurfaceKHR surface, vk::SurfaceFormatKHR format, vk::PresentModeKHR presentMode, vk::SurfaceCapabilitiesKHR capabilities, uint32_t width, uint32_t height, vk::Extent2D& ext);
    void cleanupSwapchain(vk::Device device, vk::SwapchainKHR swapchain, std::vector<vk::ImageView>& ivs, std::vector<vk::Framebuffer>& fbs, ImageDesc depth_image, ImageDesc color_image);
    void recreateSwapChain(vk::Device device, vk::SurfaceKHR surface, vk::SurfaceFormatKHR surface_format, vk::PresentModeKHR presentMode, vk::SurfaceCapabilitiesKHR capabilities, uint32_t width, uint32_t height, vk::Extent2D& ext, vk::SwapchainKHR& swapchain, ImageDesc& color_image, ImageDesc& depth_image, std::vector<vk::Image>& images, std::vector<vk::ImageView>& ivs, std::vector<vk::Framebuffer>& fbs, vk::Format format, vk::ImageAspectFlagBits aspect, vk::RenderPass rp, DeviceDesc& device_desc);

// SYNCHRONINSATION OBJECTS
	vk::Semaphore createSemaphore(vk::Device device);
	vk::Fence createFence(vk::Device device);

// BUFFER FUNCTIONS
	BufferDesc createBuffer(vk::Device device, vk::BufferUsageFlags usage, uint64_t size);
	void allocateBufferMemory(const vk::Device& device, vk::PhysicalDeviceMemoryProperties mem_properties, BufferDesc& buffer_desc, vk::MemoryPropertyFlags properties);
	void copyBuffer(vk::Device device, vk::CommandBuffer command_buffer, BufferDesc src_buffer, BufferDesc dst_buffer, uint64_t size);
	void cleanupBuffer(vk::Device& device, BufferDesc buffer_desc);

// COMMAND FUNCTIONS
	vk::CommandPool createCommandPool(const vk::Device &device, const uint32_t &queueFamilyIndex);
	vk::CommandBuffer allocateCommandBuffer(const vk::Device& device, const vk::CommandPool command_pool);
	std::array<vk::ClearValue, 2> clearColor(std::array<float, 4> color);
	void beginRenderpass(vk::CommandBuffer cmd_buffer, vk::RenderPass& renderpass, vk::Framebuffer framebuffer, vk::Extent2D extent, std::vector<vk::ClearValue> clear_values);
	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t vertex_count, vk::CommandBuffer cmd_buffer, vk::Buffer vertex_buffers[], vk::DescriptorSet descriptor_set, uint64_t offsets[]);
	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t index_count, vk::CommandBuffer cmd_buffer, vk::Buffer vertex_buffers[], vk::Buffer index_buffer, vk::DescriptorSet descriptor_set, uint64_t offsets[], uint32_t binding_count, uint32_t instance_count);
	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t vertex_count, vk::CommandBuffer cmd_buffer, BufferDesc vertex_buffer_desc);
	void recordComputeCommandBuffer(vk::Device device, vk::CommandBuffer cmd_buffer, vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, vk::DescriptorSet dsc_set, uint32_t x, uint32_t y, uint32_t z);
	void endRenderpass(vk::CommandBuffer command_buffer);
	void beginSingleTimeCommands(vk::Device device, vk::CommandBuffer& command_buffer);
	void endSingleTimeCommands(vk::Device device, vk::CommandBuffer &command_buffer, vk::CommandPool cmd_pool, vk::Queue queue);
	void submitCommand(vk::CommandBuffer command_buffer, vk::Queue queue, vk::Fence fence);

// DESCRIPTOR FUNCTIONS
	vk::DescriptorSetLayoutBinding createDescriptorSetLayoutBinding(uint8_t binding, vk::DescriptorType descriptor_type, vk::ShaderStageFlags shader_stage);
	vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device, std::vector<vk::DescriptorSetLayoutBinding> layout_bindings);
	vk::DescriptorSetLayout createComputeDescriptorSetLayout(const vk::Device& device);
	vk::DescriptorPool createImageDescriptorPool(vk::Device device, uint32_t count);
	vk::DescriptorPoolSize createDescriptorPoolSize(vk::DescriptorType descriptor_type, uint32_t count);
	vk::DescriptorPool createDescriptorPool(vk::Device device, std::vector<vk::DescriptorPoolSize> pool_sizes, uint8_t max_sets);
	vk::DescriptorPool createSsboDescriptorPool(vk::Device device, uint32_t count);
	vk::DescriptorSet allocateDescriptorSet(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout, vk::DescriptorPool descriptor_pool);
	void updateDescriptorSet(vk::Device device, vk::DescriptorSet &descriptorSet, vk::Buffer &uniformBuffer, uint64_t size, vk::ImageView textureImageView, vk::Sampler textureSampler);
	vk::DescriptorImageInfo createDescriptorImageInfo(ImageDesc image_desc, vk::Sampler sampler, vk::ImageLayout layout);
	vk::DescriptorBufferInfo createDescriptorBufferInfo(BufferDesc buffer_desc, uint64_t size);
	vk::WriteDescriptorSet createWriteDescriptorSet(vk::DescriptorSet descriptor_set, vk::DescriptorType type, vk::DescriptorBufferInfo buffer_info, uint32_t binding);
	vk::WriteDescriptorSet createWriteDescriptorSet(vk::DescriptorSet descriptor_set, vk::DescriptorType type, vk::DescriptorImageInfo image_info, uint32_t binding);

// DEVICE FUNCTIONS
	void findQueueFamilies(DeviceDesc& device_desc);
	void getPhysicalDevices(vk::Instance instance, std::vector<DeviceDesc> &device_descs);
	vk::Device createLogicalDevice(DeviceDesc device_desc, const std::vector<const char*> device_extensions, const std::vector<const char*> validation_layers, bool enable_validation_layers);
	vk::SampleCountFlagBits getMaxUsableSampleCount(vk::PhysicalDevice physical_device);

// IMAGE FUNCTIONS
    vk::Image createImage(vk::Device device, uint32_t width, uint32_t height, uint32_t mip_levels, vk::SampleCountFlagBits num_samples, vk::Format format, vk::ImageUsageFlags usage, vk::ImageType image_type, uint32_t array_layers, vk::ImageCreateFlags flags);
	vk::ImageView createImageView(const vk::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels, uint32_t layer_count, vk::ImageViewType view_type, uint32_t base);
    vk::ImageView createImageView(const vk::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels, uint32_t layer_count,vk::ImageViewType view_type);
    void allocateImageMemory(const vk::Device& device, vk::PhysicalDeviceMemoryProperties mem_properties, const vk::Image& image, vk::DeviceMemory& image_memory, vk::MemoryPropertyFlags properties);
    void createImageDesc(vk::Device device, ImageDesc& image, vk::Format format, uint32_t width, uint32_t height, vk::SampleCountFlagBits msaa_samples, vk::ImageUsageFlags usage, vk::ImageAspectFlagBits aspect, vk::ImageType image_type, vk::ImageViewType view_type, uint32_t array_layers, vk::ImageCreateFlags flags, DeviceDesc device_desc, uint32_t mip_levels);
    void cleanupImage(vk::Device device, ImageDesc image_desc);
    vk::Framebuffer createFramebuffer(const vk::Device& device, vk::ImageView swapchain_imageview, vk::ImageView color_imageview, const vk::ImageView& depth_imageview, const vk::RenderPass& rp, vk::Extent2D extent);
	vk::Framebuffer createFramebuffer(const vk::Device& device, vk::ImageView imageview, const vk::RenderPass& rp, uint32_t width, uint32_t height);
	void transitionImageLayout(vk::Device device, ImageDesc image_desc, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Format format, vk::CommandBuffer command_buffer, uint32_t mip_levels, uint32_t layer_count);
	void transitionImageLayout(vk::CommandBuffer commandBuffer, ImageDesc texture, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage);
	void copyBufferToImage(vk::Device device, BufferDesc buffer_desc, ImageDesc image_desc, uint32_t width, uint32_t height, uint32_t layer_count, vk::CommandBuffer command_buffer);
    void generateMipmaps(vk::Device device, vk::CommandBuffer command_buffer, vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, vk::FormatProperties format_properties);
    vk::Image createTextureImage(vk::Device device, const char* path, uint32_t mip_levels, vk::SampleCountFlagBits msaa_samples, vk::DeviceMemory& textureImageMemory, DeviceDesc device_desc, vk::CommandBuffer command_buffer, vk::Queue queue);
    vk::ImageView createTextureImageView(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels);
    vk::Sampler createTextureSampler(vk::Device device, vk::PhysicalDeviceProperties properties, uint32_t mip_levels);
	vk::Sampler createShadowSampler(vk::Device device, vk::PhysicalDeviceProperties properties);
	void getTextureDimensions(const char* path, int& tex_width, int &tex_height);
	stbi_uc* loadTexture(const char* path, int& width, int& height, int& channel, uint64_t& size);


//	MATRIX CALCULATIONS
	void rotateModel(glm::mat4& modelMatrix, float angle, char axis);

	void rotate(glm::mat4& model_matrix, float x, float y, float z);
	void translate(glm::mat4& model, float x, float y, float z);
	void scale(glm::mat4& model, float x, float y, float z);



// COMMON
	void computePlaneTangents(std::vector<glm::vec3>& tangents, glm::vec3* positions, glm::vec2* texCoords, uint32_t quadVertexCount);
	void fillPlaneGrid(glm::vec3* positionPtr, glm::vec2* texCoordPtr, glm::vec3* normalPtr, uint32_t* indexPtr, glm::vec3 upVector, uint32_t n, std::vector<glm::vec3>& tangents, glm::vec3* positions, glm::vec2* texCoords, uint32_t quadVertexCount);
	void computeTangents(const std::vector<glm::vec3>& positions, const std::vector<glm::vec2>& uv, const std::vector<uint32_t>& indices, std::vector<glm::vec3>& tangents);
	void fillGrid(std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uv, std::vector<uint32_t>& indices, std::vector<glm::vec3>& tangents, uint32_t stacks, uint32_t slices);
	// Function to convert equirectangular HDR to cube map
	void convertEquirectangularToCubemap(const float* hdrData, int width, int height, int faceSize, std::vector<float>& cubemapData);
}