
#pragma once

#include <set>
#include <chrono>  
#include <random>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <hdx/window.h>
#include <hdx/hdx.hpp>
#include <hdx/mvp.h>
#include <hdx/perspective_camera.h>


struct Vertex
{
	glm::vec3 position;
	glm::vec2 texcoord;
	glm::vec3 normal;
};

class Application
{
private:
	const bool enable_validation_layers = true;

	const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };
	const std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	vk::Instance instance{ nullptr };
	vk::DebugUtilsMessengerEXT debug_messenger{ nullptr };
	vk::DispatchLoaderDynamic dldi;
	vk::SurfaceKHR surface;

	std::vector<hdx::DeviceDesc> device_descs;


public:
	Application();
	~Application();
	void update(float delta_time, AppState& app_state);

	Window* window;
	const uint32_t WIDTH = 640, HEIGHT = 480;

	hdx::DeviceDesc device_desc;
	vk::Device device;
	vk::Queue queue;
	uint32_t queue_family_index;
	vk::PhysicalDeviceLimits limits;

	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;

	vk::SwapchainKHR swap_chain;
	std::vector<vk::Image> swapchain_images;
	std::vector<vk::ImageView> swapchain_imageviews;
	std::vector<vk::Framebuffer> framebuffers;
	vk::Extent2D extent;
	uint32_t swapchain_size;

	hdx::BufferDesc vb_desc, ub_desc, ib_desc, tb_desc;
	vk::Sampler sampler;
	hdx::ImageDesc texture;

	vk::ImageUsageFlags sampled_usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageViewType view_type_2d = vk::ImageViewType::e2D;
	vk::ImageType image_type_2d = vk::ImageType::e2D;
	
	std::vector<vk::VertexInputBindingDescription> binding_descriptions;
	std::vector<vk::VertexInputAttributeDescription> attribute_descriptions;

	hdx::ImageDesc color_image, depth_image;

	vk::RenderPass renderpass;

	vk::CommandPool command_pool;
	vk::CommandBuffer s_command_buffer, command_buffer;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	vk::Pipeline pipeline;
	vk::PipelineLayout pipeline_layout;

	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSetLayout _DSL;
	vk::DescriptorSet _DS;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::DescriptorBufferInfo _DBI;
	vk::DescriptorImageInfo _DII;
	std::vector<vk::DescriptorPoolSize> _DPS;
	
	vk::SampleCountFlagBits msaa_samples;
	vk::PipelineStageFlags wait_stages[2];
	uint32_t image_index;

	MVP mvp;
	PerspectiveCamera camera3D;


	
	glm::vec3 upVector = glm::vec3(0.0f, 1.0f, 0.0f);
	uint32_t n = 50;
	uint32_t vertex_count = n * n;

	glm::vec3* positions = new glm::vec3[vertex_count];
	glm::vec2* texCoords = new glm::vec2[vertex_count];
	glm::vec3* normals = new glm::vec3[vertex_count];

	std::vector<glm::vec3> tangents;

	uint32_t index_count = 6 * (n - 1) * (n - 1);   // Number of indices
	uint32_t* indices = new uint32_t[index_count];

	glm::vec3* positionPtr = positions;
	glm::vec2* texCoordPtr = texCoords;
	glm::vec3* normalPtr = normals;
	uint32_t* indexPtr = indices;

	std::vector<Vertex> vertices;
	
	/*
	std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
	{{ 0.5f, -0.5f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
	{{ 0.5f,  0.5f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f,  0.5f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
	};
	std::vector<uint32_t> indices = {
		0, 1, 2,
		2, 3, 0
	};
	uint32_t index_count = 6;
	uint32_t vertex_count = 4;
	*/
};