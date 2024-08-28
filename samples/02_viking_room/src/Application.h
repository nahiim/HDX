
#pragma once

#include <set>
#include <chrono>
#include <vector>

#include <hdx/window.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <hdx/hdx.hpp>
#include <hdx/perspective_camera.h>
#include <hdx/mvp.h>

#include <tiny_obj_loader.h>
#include <stb_image.h>

struct Vertex
{
	glm::vec3 position;
	glm::vec4 color;
	glm::vec2 tex_coord;

	bool operator==(const Vertex& other) const
	{
		return
			position == other.position &&
			tex_coord == other.tex_coord &&
			color == other.color;
	}
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
	vk::SampleCountFlagBits msaa_samples;

	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;

	vk::MemoryPropertyFlags host_VC = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

	vk::SwapchainKHR swap_chain;
	std::vector<vk::Image> swapchain_images;
	std::vector<vk::ImageView> swapchain_imageviews;
	std::vector<vk::Framebuffer> framebuffers;
	vk::Extent2D extent;
	uint32_t swapchain_size;

	hdx::ImageDesc color_image, depth_image;

	vk::DescriptorPool descriptor_pool;

	vk::DescriptorSetLayout room_DSL;
	vk::DescriptorSet room_DS;
	std::vector<vk::DescriptorSetLayoutBinding> room_DSLB;
	std::vector<vk::DescriptorPoolSize> room_DPS;
	std::vector<vk::WriteDescriptorSet> room_WDS;
	vk::Sampler room_sampler;
	vk::DescriptorImageInfo room_DII;
	vk::DescriptorBufferInfo room_DBI;
	vk::Pipeline room_pipeline;
	vk::PipelineLayout room_PL;
	hdx::ImageDesc room_texture;
	hdx::BufferDesc room_UB;
	hdx::BufferDesc room_VB;
	hdx::BufferDesc room_IB;
	hdx::BufferDesc room_TB;
	std::vector<Vertex> room_vertices;
	std::vector<uint32_t> room_indices;
	uint64_t room_VB_size, room_IB_size, room_TB_size;
	MVP room_mvp;
	MVP mvp;

	std::vector<vk::VertexInputBindingDescription> room_VIBD;
	std::vector<vk::VertexInputAttributeDescription> room_VIAD;

	vk::ImageUsageFlags sampled_usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageUsageFlags color_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment;
	vk::ImageUsageFlags depth_usage_flags = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment;
	vk::ImageUsageFlags cubemap_usage_flags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageViewType view_type_2d = vk::ImageViewType::e2D;
	vk::ImageViewType view_type_cube = vk::ImageViewType::eCube;
	vk::ImageType image_type_2d = vk::ImageType::e2D;

	vk::RenderPass renderpass;

	vk::CommandPool command_pool;
	vk::CommandBuffer command_buffer, s_command_buffer, asd;

	vk::Semaphore image_available_semaphore;
	vk::Semaphore render_finished_semaphore;
	vk::Fence in_flight_fence;
	
	PerspectiveCamera camera3D;
	uint32_t current_frame = 0;
	uint32_t image_index;
	vk::PipelineStageFlags wait_stages[2];
};