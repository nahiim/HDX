
#pragma once

#include <set>
#include <chrono>  
#include <random>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <hdx/window.h>
#include <hdx/hdx.hpp>


struct Particle
{
	glm::vec2 position;
	glm::vec2 velocity;
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
	vk::CommandBuffer g_command_buffer, c_command_buffer;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Semaphore compute_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	vk::Pipeline graphics_pipeline, compute_pipeline;
	vk::PipelineLayout g_pipeline_layout, c_pipeline_layout;

	hdx::BufferDesc ssb, ubo;
	float dt;

	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSetLayout _DSL;
	vk::DescriptorSet _DS;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::DescriptorBufferInfo _DBI_u;
	vk::DescriptorBufferInfo _DBI_s, _DBI_s2;
	std::vector<vk::DescriptorPoolSize> pool_sizes;

	const uint32_t PARTICLE_COUNT = 8192;
	std::vector<Particle> particles;
	vk::SampleCountFlagBits msaa_samples;

	float last_frame_time = 0.0f;

	vk::PipelineStageFlags wait_stages[2];
};