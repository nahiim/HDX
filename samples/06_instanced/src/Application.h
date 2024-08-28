
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

struct VPMatrices
{
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec4 cam_pos;
};
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

struct InstanceData
{
	glm::mat4 modelMatrix = glm::mat4(1.0f);
	float texture_id;
};



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

	VPMatrices vp;

	std::vector<vk::VertexInputBindingDescription> binding_descriptions, cube_binding_descriptions;
	std::vector<vk::VertexInputAttributeDescription> attribute_descriptions, cube_attribute_descriptions;

	hdx::ImageDesc color_image, depth_image;
	vk::RenderPass renderpass;

	vk::CommandPool command_pool;
	vk::CommandBuffer command_buffer, s_command_buffer;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;

	uint32_t current_frame = 0;

	hdx::ImageDesc cube_texture;
	hdx::BufferDesc cube_vb, sphere_vb, cube_tb, sphere_ib, ub, sphere_tb, light_ub, instance_b;
	vk::Sampler cube_sampler, sphere_sampler;
	hdx::ImageDesc sphere_texture;
	Light light;
	glm::mat4 model;


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

	vk::Pipeline pipeline, sphere_pipeline;
	vk::PipelineLayout pipeline_layout, sphere_pl;

	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSetLayout _DSL, _DSL0;
	vk::DescriptorSet _DS, _DS0;
	std::vector<vk::DescriptorSetLayoutBinding> _DSLB, _DSLB0;
	std::vector<vk::DescriptorPoolSize> pool_sizes;
	std::vector<vk::WriteDescriptorSet> _WDS;
	vk::DescriptorImageInfo _DII, _DII_sphere;
	vk::DescriptorBufferInfo _DBI, _DBI_light, _DBI_transform;

	vk::ImageUsageFlags sampled_usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageUsageFlags color_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment;
	vk::ImageUsageFlags depth_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment;
	vk::ImageUsageFlags cubemap_usage_flags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageViewType view_type_2d = vk::ImageViewType::e2D;
	vk::ImageViewType view_type_cube = vk::ImageViewType::eCube;
	vk::ImageType image_type_2d = vk::ImageType::e2D;

	uint32_t instance_count = 3;
	std::vector<InstanceData> instances;

	std::vector<Vertex> vertices;

	uint32_t stacks = 29, slices = 29;
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> uv;
	std::vector<uint32_t> indices;
	std::vector<glm::vec3> tangents;
	uint32_t vertex_count = (slices + 1) * (stacks + 1);
	uint32_t index_count = stacks * slices * 6;

	int sphere_img_width, sphere_img_height, sphere_img_channel;
	uint64_t sphere_img_size;
	std::vector<uint8_t> sphere_pixels;

	std::vector<float> cube_coords = {
		// Face 1 (Front)
		-1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left
		 1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-right
		 1.0f,  1.0f, -1.0f, 1.0f,  // Top-right
		 1.0f,  1.0f, -1.0f, 1.0f,  // Top-right (repeat)
		-1.0f,  1.0f, -1.0f, 1.0f,  // Top-left
		-1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left (repeat)

		// Face 2 (Back)
		-1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-left
		 1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-right
		 1.0f,  1.0f,  1.0f, 1.0f,  // Top-right
		 1.0f,  1.0f,  1.0f, 1.0f,  // Top-right (repeat)
		-1.0f,  1.0f,  1.0f, 1.0f,  // Top-left
		-1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-left (repeat)

		// Face 3 (Left)
		-1.0f,  1.0f,  1.0f, 1.0f,  // Top-right
		-1.0f,  1.0f, -1.0f, 1.0f,  // Top-left
		-1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left
		-1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left (repeat)
		-1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-right
		-1.0f,  1.0f,  1.0f, 1.0f,  // Top-right (repeat)

		// Face 4 (Right)
		 1.0f,  1.0f,  1.0f, 1.0f,  // Top-right
		 1.0f,  1.0f, -1.0f, 1.0f,  // Top-left
		 1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left
		 1.0f, -1.0f, -1.0f, 1.0f,  // Bottom-left (repeat)
		 1.0f, -1.0f,  1.0f, 1.0f,  // Bottom-right
		 1.0f,  1.0f,  1.0f, 1.0f,  // Top-right (repeat)

		 // Face 5 (Bottom)
		 -1.0f, -1.0f, -1.0f, 1.0f, // Bottom-left
		  1.0f, -1.0f, -1.0f, 1.0f, // Bottom-right
		  1.0f, -1.0f,  1.0f, 1.0f, // Top-right
		  1.0f, -1.0f,  1.0f, 1.0f, // Top-right (repeat)
		 -1.0f, -1.0f,  1.0f, 1.0f, // Top-left
		 -1.0f, -1.0f, -1.0f, 1.0f, // Bottom-left (repeat)

		 // Face 6 (Top)
		 -1.0f,  1.0f, -1.0f, 1.0f, // Top-left
		  1.0f,  1.0f, -1.0f, 1.0f, // Top-right
		  1.0f,  1.0f,  1.0f, 1.0f, // Bottom-right
		  1.0f,  1.0f,  1.0f, 1.0f, // Bottom-right (repeat)
		 -1.0f,  1.0f,  1.0f, 1.0f, // Bottom-left
		 -1.0f,  1.0f, -1.0f, 1.0f, // Top-left (repeat)
	};

	PerspectiveCamera camera3D;

	vk::SampleCountFlagBits msaa_samples;	vk::PipelineStageFlags wait_stages[2];

};