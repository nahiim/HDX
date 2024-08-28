
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
	glm::vec4 position;
	glm::vec4 normal;
	glm::vec4 uv;
	glm::vec4 tangent;
};
struct Light
{
	glm::vec4 position;
	glm::vec4 color;
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
	
	std::vector<vk::VertexInputBindingDescription> binding_descriptions;
	std::vector<vk::VertexInputAttributeDescription> attribute_descriptions;

	hdx::ImageDesc color_image, depth_image;

	vk::RenderPass renderpass;

	vk::CommandPool command_pool;
	vk::CommandBuffer command_buffer, s_cmd;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	vk::Pipeline pipeline;
	vk::PipelineLayout pipeline_layout;

	hdx::BufferDesc vb, ib, ub, diffuse_tb, normal_tb, light_ub;
	vk::Sampler diffuse_sampler, normal_sampler;
	hdx::ImageDesc diffuse_image_desc, normal_image_desc;
	MVP mvp;
	Light light;
	PerspectiveCamera camera3D;

	int diffuse_width, diffuse_height, normal_width, normal_height, diffuse_channel, normal_channel;
	uint64_t diffuse_size, normal_size;
	
	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSetLayout _DSL;
	vk::DescriptorSet _DS;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::DescriptorBufferInfo _DBI_u, _DBI_light;
	vk::DescriptorImageInfo _DII_diffuse, _DII_normal;
	std::vector<vk::DescriptorPoolSize> pool_sizes;

	std::vector<Vertex> vertices;

	uint32_t stacks = 29, slices = 29;
	
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> uv;
	std::vector<uint32_t> indices;
	std::vector<glm::vec3> tangents;

	uint32_t vertex_count = (slices+1) * (stacks+1);
	uint32_t index_count = stacks * slices * 6;

	float last_frame_time = 0.0f;
	vk::SampleCountFlagBits msaa_samples;
	vk::PipelineStageFlags wait_stages[2];
};