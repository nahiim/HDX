
#include "Application.h"


std::string GetScreenshotFilename(uint32_t& screenshot_index)
{
	char buffer[256];
	snprintf(buffer, sizeof(buffer), "agenerated/screenshot_%03d.png", screenshot_index++);

	return std::string(buffer);
}

bool screenshot(vk::Device device, hdx::BufferDesc ssb, int width, int height, uint32_t& screenshot_index)
{
	std::vector<uint8_t> imageData(4 * width * height);
	hdx::copyFromDevice(device, ssb, imageData.data(), 4 * width * height);

	std::string filename = GetScreenshotFilename(screenshot_index);
	int success = stbi_write_png(filename.c_str(), width, height, 4, imageData.data(), width * 4);
	return success != 0;
}



Application::Application()
{
	float aspectRatio = float(WIDTH) / float(HEIGHT);
	camera = PerspectiveCamera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
	mvp.projection = camera.getProjectionMatrix();
	mvp.view = camera.getViewMatrix();
	mvp.model = glm::mat4(1.0f);

	window = new Window("Raytracer(Compute)", WIDTH, HEIGHT);
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
	device_desc.features.shaderFloat64 = VK_TRUE;

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
	std::cout << "Shared Memory Size: " << limits.maxComputeSharedMemorySize << std::endl;
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

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	ray_cam = RayCamera(glm::vec3(-5.0f, 3.0f, 0.0f), glm::vec3(1.0f, 0.0f, .0f));

	ray_cam_ubo = { fr_id, Nx, Ny, ray_cam.position, ray_cam.direction, ray_cam.up, ray_cam.right };
	image_width = ray_cam_ubo.Nx, image_height = ray_cam_ubo.Ny;

	float* image_data = stbi_loadf("res/night.hdr", &hdr_width, &hdr_height, &hdr_channels, 0);
	if (!image_data) {
		throw std::runtime_error("Failed to load texture image!");
	}
	uint64_t image_size = hdr_width * hdr_height * sizeof(float) * 4;

	// Allocate RGBA float buffer
	std::vector<float> rgbaData(hdr_width * hdr_height * 4);

	for (int i = 0; i < hdr_width * hdr_height; ++i) {
		rgbaData[i * 4 + 0] = image_data[i * hdr_channels + 0]; // R
		rgbaData[i * 4 + 1] = image_data[i * hdr_channels + 1]; // G
		rgbaData[i * 4 + 2] = image_data[i * hdr_channels + 2]; // B
		rgbaData[i * 4 + 3] = 1.0f;                   // A
	}

	sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, hdr_texture, vk::Format::eR32G32B32A32Sfloat, hdr_width, hdr_height, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);
	image_info = hdx::createDescriptorImageInfo(hdr_texture, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);

	hdr_tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, image_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, hdr_tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, hdr_tb, rgbaData.data(), image_size);

	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::transitionImageLayout(device, hdr_texture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
		hdx::copyBufferToImage(device, hdr_tb, hdr_texture, hdr_width, hdr_height, 1, command_buffer);
		hdx::transitionImageLayout(device, hdr_texture.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);

	hdx::createImageDesc(device, input_texture, vk::Format::eR8G8B8A8Srgb, image_width, image_height, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);

	ub_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub_desc, &mvp, sizeof(MVP));
	mvp_info = hdx::createDescriptorBufferInfo(ub_desc, sizeof(MVP));

	ubo = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UBO));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ubo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ubo, &ray_cam_ubo, sizeof(UBO));
	ubo_info = hdx::createDescriptorBufferInfo(ubo, sizeof(UBO));

	ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, sizeof(glm::vec4) * Nx * Ny);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	ssb_info = hdx::createDescriptorBufferInfo(ssb, sizeof(glm::vec4) * Nx * Ny);

	accum_ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, Nx * Ny * sizeof(glm::vec4));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, accum_ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	accum_ssb_info = hdx::createDescriptorBufferInfo(accum_ssb, Nx * Ny * sizeof(glm::vec4));

	bright_ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, Nx * Ny * sizeof(glm::vec4));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, bright_ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	bright_ssb_info = hdx::createDescriptorBufferInfo(bright_ssb, Nx * Ny * sizeof(glm::vec4));

	hdr_ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, Nx * Ny * sizeof(glm::vec4));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, hdr_ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdr_ssb_info = hdx::createDescriptorBufferInfo(hdr_ssb, Nx * Ny * sizeof(glm::vec4));

	uint64_t rays_size = image_width * image_height * sizeof(Ray);
	ray_ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, rays_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ray_ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	ray_ssb_info = hdx::createDescriptorBufferInfo(ray_ssb, rays_size);


	pool_sizes = {
		hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 5)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 1);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(7, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute)
	};
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);

	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, mvp_info, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, image_info, 1),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, ssb_info, 2),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, ubo_info, 3),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, ray_ssb_info, 4),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, accum_ssb_info, 5),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, bright_ssb_info, 6),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, hdr_ssb_info, 7)
	};
	device.updateDescriptorSets(8, _WDS.data(), 0, nullptr);


	rtx_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	rtx_pipeline = hdx::createComputePipeline(device, _DSL, rtx_pipeline_layout, "res/shaders/rtx.comp.spv");
	h_blur = hdx::createComputePipeline(device, _DSL, rtx_pipeline_layout, "res/shaders/horizontal_blur.comp.spv");
	v_blur = hdx::createComputePipeline(device, _DSL, rtx_pipeline_layout, "res/shaders/vertical_blur.comp.spv");
	composite = hdx::createComputePipeline(device, _DSL, rtx_pipeline_layout, "res/shaders/composite_pass.comp.spv");

	uint32_t Nx = image_width, Ny = image_height;
	Dimension block_dimension = { 32, 32, 1 };
	grid_dimension = {
		(Nx + block_dimension.x - 1) / block_dimension.x,
		(Ny + block_dimension.y - 1) / block_dimension.y,
									(1)
	};


	// RayTracing computation
	hdx::recordComputeCommandBuffer(device, command_buffer, rtx_pipeline, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer
	
	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::transitionImageLayout(device, input_texture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
		hdx::copyBufferToImage(device, ssb, input_texture, image_width, image_height, 1, command_buffer);
	hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);


	//screenshot(device, ssb, Nx, Ny, screenshot_index);


	camera.translate(0, 0, 3.77);
	//camera.rotate(-2000, 0, 0);
}


void Application::update(float delta_time, AppState& app_state)
{
	// RayTracing computation
	hdx::recordComputeCommandBuffer(device, command_buffer, rtx_pipeline, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// Horizontal gaussian blur
	hdx::recordComputeCommandBuffer(device, command_buffer, h_blur, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// Vertical Gaussian blur
	hdx::recordComputeCommandBuffer(device, command_buffer, v_blur, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// Composite pass
	hdx::recordComputeCommandBuffer(device, command_buffer, composite, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::copyBufferToImage(device, ssb, input_texture, image_width, image_height, 1, command_buffer);
		hdx::transitionImageLayout(device, input_texture.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);

	fr_id++;
	if (cameraMoved)
	{
		// Only reset accumulation buffer after camera has moved
		fr_id = 1;  // Start fresh accumulation after reset
		cameraMoved = false;  // Reset flag until next movement
	}

	mvp.view = camera.getViewMatrix();
	hdx::copyToDevice(device, ub_desc, &mvp, sizeof(MVP));

	ray_cam_ubo = { fr_id, Nx, Ny, ray_cam.position, ray_cam.direction, ray_cam.up, ray_cam.right };
	hdx::copyToDevice(device, ubo, &ray_cam_ubo, sizeof(UBO));
	float rv = 0.02f * delta_time;
	float mv = 0.002f * delta_time;
	if (Input::GetKey(Input::KEY_I))
	{
		ray_cam.rotate(-rv, 0, 0);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_J))
	{
		ray_cam.rotate(0, rv, 0);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_K))
	{
		ray_cam.rotate(rv, 0, 0);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_L))
	{
		ray_cam.rotate(0, -rv, 0);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_W))
	{
		ray_cam.translate(0, 0, mv);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_A))
	{
		ray_cam.translate(-mv, 0, 0);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_S))
	{
		ray_cam.translate(0, 0, -mv);
		cameraMoved = true;
	}
	if (Input::GetKey(Input::KEY_D))
	{
		ray_cam.translate(mv, 0, 0);
		cameraMoved = true;
	}

	if (Input::GetKey(Input::KEY_SPACE))
	{
		screenshot(device, ssb, Nx, Ny, screenshot_index);
	}

	if (Input::GetKey(Input::KEY_ESCAPE))
	{
		app_state.running = false;
	}

	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);
	device.resetFences({ in_flight_fence });

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
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });

	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::transitionImageLayout(device, swapchain_images[image_index], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
		hdx::copyImage(command_buffer,	input_texture.image, swapchain_images[image_index], vk::Extent3D{ WIDTH, HEIGHT, 1 }, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
	hdx::endSingleTimeCommands(image_available_semaphore, render_finished_semaphore, in_flight_fence, command_buffer, queue);

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

	hdx::cleanupBuffer(device, ub_desc);
	hdx::cleanupBuffer(device, ssb);
	hdx::cleanupBuffer(device, accum_ssb);
	hdx::cleanupBuffer(device, ray_ssb);
	hdx::cleanupBuffer(device, bright_ssb);
	hdx::cleanupBuffer(device, hdr_ssb);
	hdx::cleanupBuffer(device, ubo);
	hdx::cleanupImage(device, input_texture);
	hdx::cleanupBuffer(device, hdr_tb);
	hdx::cleanupImage(device, hdr_texture);
	hdx::cleanupImage(device, output_texture);
	device.destroySampler(sampler);

	device.destroyPipeline(rtx_pipeline);
	device.destroyPipeline(h_blur);
	device.destroyPipeline(v_blur);
	device.destroyPipeline(composite);
	device.destroyPipelineLayout(rtx_pipeline_layout);

	device.destroyRenderPass(renderpass);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);

	device.destroySemaphore(render_finished_semaphore);
	device.destroySemaphore(image_available_semaphore);
	device.destroyFence(in_flight_fence);

	device.destroyCommandPool(command_pool);

	device.destroy();
	instance.destroySurfaceKHR(surface);

	if (enable_validation_layers) { instance.destroyDebugUtilsMessengerEXT(debug_messenger, nullptr, dldi); }

	instance.destroy();	std::cout << "Application DESTROYED!!\n";

	delete window;
}