
#include "Application.h"

std::vector<Particle> generateParticles(uint32_t width, uint32_t height, uint32_t count)
{
	// Initialize particles
	std::default_random_engine rndEngine((unsigned)time(nullptr));
	std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

	// Initial particle positions on a circle
	std::vector<Particle> particles(count);
	for (auto& particle : particles)
	{
		float r = 0.5f * sqrt(rndDist(rndEngine));
		float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
		float x = r * cos(theta) * height / width;
		float y = r * sin(theta);
		particle.position = glm::vec2(x, y);
		particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
		particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
	}

	return particles;
}


Application::Application()
{
	window = new Window("HYDROXY - Particles(Compute)", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "Particles", enable_validation_layers, validation_layers);
	dldi = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);
	hdx::createDebugMessenger(debug_messenger, instance, dldi);
	window->createSurface(surface, instance);

	hdx::getPhysicalDevices(instance, device_descs);
	device_desc = device_descs[0];
	hdx::findQueueFamilies(device_desc);

	device = hdx::createLogicalDevice(device_desc, device_extensions, validation_layers, enable_validation_layers);

	for (uint32_t i = 0; i < device_desc.queue_families.size(); i++)
	{
		if (device_desc.queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			queue_family_index = i;
			queue = device.getQueue(queue_family_index, 0);
			break;  // Break the loop once a queue is found
		}
	}

	device_desc.physical_device.getMemoryProperties(&device_desc.memory_properties);
	device_desc.physical_device.getProperties(&device_desc.properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;
	msaa_samples = hdx::getMaxUsableSampleCount(device_desc.physical_device);
	device_desc.physical_device.getFormatProperties(vk::Format::eR8G8B8A8Srgb, &device_desc.format_properties);
	limits = device_desc.properties.limits;
	std::cout << "maxComputeWorkGroupCount: "
		<< limits.maxComputeWorkGroupCount[0] << ", "
		<< limits.maxComputeWorkGroupCount[1] << ", "
		<< limits.maxComputeWorkGroupCount[2] << std::endl;

	std::cout << "maxComputeWorkGroupInvocations: "
		<< limits.maxComputeWorkGroupInvocations << std::endl;

	std::cout << "maxComputeWorkGroupSize: "
		<< limits.maxComputeWorkGroupSize[0] << ", "
		<< limits.maxComputeWorkGroupSize[1] << ", "
		<< limits.maxComputeWorkGroupSize[2] << std::endl;
	device_desc.features.samplerAnisotropy = VK_TRUE;

	capabilities = device_desc.physical_device.getSurfaceCapabilitiesKHR(surface);
	formats = device_desc.physical_device.getSurfaceFormatsKHR(surface);
	presentModes = device_desc.physical_device.getSurfacePresentModesKHR(surface);

	swap_chain = hdx::createSwapchain(device, surface, vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities, WIDTH, HEIGHT, extent);
	swapchain_images = device.getSwapchainImagesKHR(swap_chain);
	swapchain_size = swapchain_images.size();

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}
	
	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Particle), vk::VertexInputRate::eVertex) };
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)));

	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute));
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);

	graphics_pipeline = hdx::createGraphicsPipeline(device, g_pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::ePointList, extent);
	compute_pipeline = hdx::createComputePipeline(device, _DSL, c_pipeline_layout, "res/shaders/shader.comp.spv");

	command_pool = hdx::createCommandPool(device, queue_family_index);
	g_command_buffer = hdx::allocateCommandBuffer(device, command_pool);
	c_command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	particles = generateParticles(WIDTH, HEIGHT, PARTICLE_COUNT);

	uint64_t sz = sizeof(Particle) * particles.size();
	ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer, sz);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ssb, particles.data(), sz);
	
	ubo = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(float));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ubo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ubo, &dt, sizeof(float));

	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eStorageBuffer, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eStorageBuffer, swapchain_size));
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 1);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);
	_DBI_u = hdx::createDescriptorBufferInfo(ubo, sizeof(float));
	_DBI_s = hdx::createDescriptorBufferInfo(ssb, sz);
	_DBI_s2 = hdx::createDescriptorBufferInfo(ssb, sz);
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_u, 0));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, _DBI_s, 1));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, _DBI_s2, 2));
	device.updateDescriptorSets(3, _WDS.data(), 0, nullptr);

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	compute_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
}


void Application::update(float delta_time, AppState& app_state)
{
	hdx::copyToDevice(device, ubo, &delta_time, sizeof(float));

	if (Input::GetKey(Input::KEY_ESCAPE))
	{
		app_state.running = false;
	}

	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);
	device.resetFences({ in_flight_fence });

	c_command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });
	hdx::recordComputeCommandBuffer(device, c_command_buffer, compute_pipeline, c_pipeline_layout, _DS, PARTICLE_COUNT/256, 1, 1);

	vk::SubmitInfo submit_info{};
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &c_command_buffer;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &compute_finished_semaphore;
	queue.submit(submit_info, in_flight_fence);

	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);

	uint32_t image_index;
	vk::Result acquire_result = device.acquireNextImageKHR(swap_chain, UINT64_MAX, image_available_semaphore, nullptr, &image_index); 

	if (acquire_result == vk::Result::eErrorOutOfDateKHR)
	{
		hdx::recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eFifo, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}

	device.resetFences({ in_flight_fence });
	g_command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};
	hdx::beginRenderpass(g_command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(graphics_pipeline, g_pipeline_layout, PARTICLE_COUNT, g_command_buffer, ssb);
	hdx::endRenderpass(g_command_buffer);

	vk::Semaphore wait_semaphores[] = { compute_finished_semaphore, image_available_semaphore };

	submit_info = 0;
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.waitSemaphoreCount = 2;
	submit_info.pWaitSemaphores = wait_semaphores;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &render_finished_semaphore;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &g_command_buffer;
	queue.submit({ submit_info }, in_flight_fence);

	vk::SwapchainKHR swapChains[] = { swap_chain };
	vk::PresentInfoKHR present_info;
	present_info.sType = vk::StructureType::ePresentInfoKHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapChains;
	present_info.pImageIndices = &image_index; // Use the acquired image index

	vk::Result present_result;
	try
	{
		present_result = queue.presentKHR(present_info);
	}
	catch (vk::OutOfDateKHRError error)
	{
		recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eFifo, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}

	current_frame = (current_frame + 1) % swapchain_size;

	queue.waitIdle();
	device.waitIdle();
}



Application::~Application()
{
	cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	hdx::cleanupBuffer(device, ssb);
	hdx::cleanupBuffer(device, ubo);

	device.destroyPipeline(compute_pipeline);
	device.destroyPipelineLayout(c_pipeline_layout);
	device.destroyPipeline(graphics_pipeline);
	device.destroyPipelineLayout(g_pipeline_layout);

	device.destroyRenderPass(renderpass);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);

	device.destroySemaphore(render_finished_semaphore);
	device.destroySemaphore(image_available_semaphore);
	device.destroySemaphore(compute_finished_semaphore);
	device.destroyFence(in_flight_fence);

	device.destroyCommandPool(command_pool);

	device.destroy();
	instance.destroySurfaceKHR(surface);

	if (enable_validation_layers) { instance.destroyDebugUtilsMessengerEXT(debug_messenger, nullptr, dldi); }

	instance.destroy();	std::cout << "Application DESTROYED!!\n";

	delete window;
}