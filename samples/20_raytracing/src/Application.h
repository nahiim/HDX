
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

#include "ray_camera.h"

struct UBO
{
	uint32_t frame_index;
	uint32_t Nx;
	uint32_t Ny;

	alignas(16)glm::vec3 ray_origin;
	alignas(16)glm::vec3 cam_direction;
	alignas(16)glm::vec3 cam_up;
	alignas(16)glm::vec3 cam_right;
};

struct Dimension
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
};

struct Vertex
{
	glm::vec3 position;
	glm::vec2 texcoord;
	glm::vec3 normal;
};

struct Ray
{
	alignas(16) glm::vec3 origin;
	alignas(16) glm::vec3 direction;
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

	uint32_t Nx = 1024;
	uint32_t Ny = Nx;


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
	vk::CommandBuffer command_buffer;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	vk::Pipeline graphics_pipeline, rtx_pipeline;
	vk::PipelineLayout graphics_pipeline_layout, rtx_pipeline_layout;

	hdx::BufferDesc vb_desc, ub_desc, ib_desc, ubo, ssb, accum_ssb, ray_ssb;
	float dt;

	vk::DescriptorPool descriptor_pool;
	std::vector<vk::DescriptorPoolSize> pool_sizes;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::DescriptorSetLayout _DSL;
	vk::DescriptorSet _DS;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB;
	vk::DescriptorBufferInfo mvp_info, ssb_info, ray_ssb_info, ubo_info, accum_ssb_info;
	vk::DescriptorImageInfo image_info;

	uint32_t image_width, image_height, bytes_per_pixel;
	hdx::ImageDesc input_texture, output_texture;
	vk::Sampler sampler;
	vk::ImageUsageFlags sampled_usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageViewType view_type_2d = vk::ImageViewType::e2D;
	vk::ImageType image_type_2d = vk::ImageType::e2D;
	vk::SampleCountFlagBits msaa_samples;

	float last_frame_time = 0.0f;

	vk::PipelineStageFlags wait_stages[2];

	
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

	MVP mvp;
	UBO ray_cam_ubo;
	PerspectiveCamera camera;
	RayCamera ray_cam;

	Dimension grid_dimension;
	uint8_t* out_image;

	uint32_t fr_id = 1;

	uint8_t* zero_bf;
	bool cameraMoved = false;
	bool reset = false;
	uint32_t screenshot_index = 0;
};