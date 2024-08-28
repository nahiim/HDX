
#pragma once

#include <set>
#include <chrono>  
#include <random>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <hdx/window.h>
#include <hdx/perspective_camera.h>
#include <hdx/hdx.hpp>
#include <hdx/mvp.h>

class Application
{
private:
	const bool enable_validation_layers = true;

	const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };
	const std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

	//instance-related variables
	vk::Instance instance{ nullptr };
	vk::DebugUtilsMessengerEXT debug_messenger{ nullptr };
	vk::DispatchLoaderDynamic dldi;
	vk::SurfaceKHR surface;

	// Will store all GPUs on the computer
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
	vk::CommandBuffer command_buffer, s_command_buffer;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	hdx::ImageDesc texture;
	hdx::BufferDesc vb, tb, ub;
	vk::Sampler sampler;

	MVP mvp;


	// Paths to your cubemap face images
	std::vector<std::string> faces = {
		"res/textures/sky/right.jpg",
		"res/textures/sky/left.jpg",
		"res/textures/sky/top.jpg",
		"res/textures/sky/bottom.jpg",
		"res/textures/sky/front.jpg",
		"res/textures/sky/back.jpg"
	};
	int image_width, image_height, image_channels;
	std::vector<uint8_t> all_pixels;

	vk::Pipeline pipeline;
	vk::PipelineLayout pipeline_layout;

	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSetLayout _DSL;
	vk::DescriptorSet _DS;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB;
	std::vector<vk::DescriptorPoolSize> pool_sizes;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::Sampler _sampler;
	vk::DescriptorImageInfo _DII;
	vk::DescriptorBufferInfo _DBI;

	vk::ImageUsageFlags sampled_usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageUsageFlags color_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment;
	vk::ImageUsageFlags depth_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment;
	vk::ImageUsageFlags cubemap_usage_flags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageViewType view_type_2d = vk::ImageViewType::e2D;
	vk::ImageViewType view_type_cube = vk::ImageViewType::eCube;
	vk::ImageType image_type_2d = vk::ImageType::e2D;

	std::vector<float> vertices = {
		// Face 1 (Front)
		-1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left
		 1.0f, -1.0f, -1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Bottom-right
		 1.0f,  1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-right
		 1.0f,  1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-right (repeat)
		-1.0f,  1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Top-left
		-1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left (repeat)

		// Face 2 (Back)
		-1.0f, -1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left
		 1.0f, -1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Bottom-right
		 1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-right
		 1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-right (repeat)
		-1.0f,  1.0f,  1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Top-left
		-1.0f, -1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left (repeat)

		// Face 3 (Left)
		-1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right
		-1.0f,  1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-left
		-1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left
		-1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left (repeat)
		-1.0f, -1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-right
		-1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right (repeat)

		// Face 4 (Right)
		 1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right
		 1.0f,  1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-left
		 1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left
		 1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left (repeat)
		 1.0f, -1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-right
		 1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right (repeat)

		 // Face 5 (Bottom)
		 -1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left
		  1.0f, -1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Bottom-right
		  1.0f, -1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right
		  1.0f, -1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Top-right (repeat)
		 -1.0f, -1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Top-left
		 -1.0f, -1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Bottom-left (repeat)

		 // Face 6 (Top)
		 -1.0f,  1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Top-left
		  1.0f,  1.0f, -1.0f, 1.0f,   1.0f, 1.0f, 0.0f, 0.0f, // Top-right
		  1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Bottom-right
		  1.0f,  1.0f,  1.0f, 1.0f,   1.0f, 0.0f, 0.0f, 0.0f, // Bottom-right (repeat)
		 -1.0f,  1.0f,  1.0f, 1.0f,   0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left
		 -1.0f,  1.0f, -1.0f, 1.0f,   0.0f, 1.0f, 0.0f, 0.0f, // Top-left (repeat)
	};




	PerspectiveCamera camera3D;

	vk::SampleCountFlagBits msaa_samples;	vk::PipelineStageFlags wait_stages[2];

};