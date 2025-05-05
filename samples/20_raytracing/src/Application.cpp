
#include "Application.h"


std::string GetScreenshotFilename(uint32_t& screenshot_index)
{
	char buffer[256];
	snprintf(buffer, sizeof(buffer), "agenerated/screenshot_%03d.png", screenshot_index++);

	return std::string(buffer);
}

bool screenshot(vk::Device device, hdx::BufferDesc ssb, int width, int height, uint32_t& screenshot_index)
{
	std::vector<glm::vec4> imageData(width * height);
	hdx::copyFromDevice(device, ssb, imageData.data(), sizeof(glm::vec4) * width * height);

	std::vector<uint8_t> rgba8(width * height * 4);
	for (int i = 0; i < width * height; ++i) {
		const glm::vec4& px = imageData[i];
		rgba8[i * 4 + 0] = static_cast<uint8_t>(std::clamp(px.r, 0.0f, 1.0f) * 255.0f);
		rgba8[i * 4 + 1] = static_cast<uint8_t>(std::clamp(px.g, 0.0f, 1.0f) * 255.0f);
		rgba8[i * 4 + 2] = static_cast<uint8_t>(std::clamp(px.b, 0.0f, 1.0f) * 255.0f);
		rgba8[i * 4 + 3] = static_cast<uint8_t>(std::clamp(px.a, 0.0f, 1.0f) * 255.0f);
	}

	std::vector<uint8_t> flippedRgba8(width * height * 4);
	for (int y = 0; y < height; ++y) {
		int srcRow = y;
		int dstRow = height - 1 - y;
		std::memcpy(&flippedRgba8[dstRow * width * 4], &rgba8[srcRow * width * 4], width * 4);
	}

	std::string filename = GetScreenshotFilename(screenshot_index);
	int success = stbi_write_png(filename.c_str(), width, height, 4, flippedRgba8.data(), width * 4);
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

	VkInstance raw_instance;

	void* inas = reinterpret_cast<void*>(raw_instance);
	raw_instance = reinterpret_cast<VkInstance>(inas);

	hdx::getPhysicalDevices(raw_instance, device_descs);
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

	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) };
	attribute_descriptions = { (
		hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0)),
		hdx::getAttributeDescription(0, 1, vk::Format::eR32G32Sfloat, offsetof(Vertex, texcoord)),
		hdx::getAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)
	) };

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	bytes_per_pixel = sizeof(glm::vec4);

	ray_cam = RayCamera(glm::vec3(-5.0f, 3.0f, 0.0f), glm::vec3(1.0f, 0.0f, .0f));

	ray_cam_ubo = { fr_id, Nx, Ny, ray_cam.position, ray_cam.direction, ray_cam.up, ray_cam.right };
	image_width = ray_cam_ubo.Nx, image_height = ray_cam_ubo.Ny;
	uint64_t image_size = static_cast<uint64_t>(image_width) * image_height * bytes_per_pixel;

	hdx::createImageDesc(device, input_texture, vk::Format::eR32G32B32A32Sfloat, image_width, image_height, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);
	sampler = hdx::createTextureSampler(device, device_desc.properties, 0);
	image_info = hdx::createDescriptorImageInfo(input_texture, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);

	vb_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, sizeof(Vertex) * vertex_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, vb_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, vb_desc, vertices.data(), sizeof(Vertex) * vertex_count);

	ib_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ib_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ib_desc, indices.data(), sizeof(uint32_t) * index_count);

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
	zero_bf = (uint8_t*)malloc(Nx * Ny * sizeof(glm::vec4));

	uint64_t rays_size = image_width * image_height * sizeof(Ray);
	ray_ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, rays_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ray_ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	ray_ssb_info = hdx::createDescriptorBufferInfo(ray_ssb, rays_size);


	pool_sizes = {
		hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 1);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute)
	};
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);

	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, mvp_info, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, image_info, 1),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, ssb_info, 2),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, ubo_info, 3),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, ray_ssb_info, 4),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, accum_ssb_info, 5)
	};
	device.updateDescriptorSets(6, _WDS.data(), 0, nullptr);


	rtx_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	rtx_pipeline = hdx::createComputePipeline(device, _DSL, rtx_pipeline_layout, "res/shaders/rtx.comp.spv");

	uint32_t Nx = image_width, Ny = image_height;
	Dimension block_dimension = { 32, 32, 1 };
	grid_dimension = { (Nx + block_dimension.x - 1) / block_dimension.x,
								 (Ny + block_dimension.y - 1) / block_dimension.y,
															(1)						};


	// RayTracing computation
	hdx::recordComputeCommandBuffer(device, command_buffer, rtx_pipeline, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	
	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
		hdx::copyBufferToImage(device, ssb, input_texture, image_width, image_height, 1, command_buffer);
		hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);

	if (image_height == image_width)
		std::cout << "\n\n" << bytes_per_pixel << "\n\n";

	screenshot(device, ssb, Nx, Ny, screenshot_index);

	// Free the image memory
//	stbi_image_free(pixels);

	graphics_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	graphics_pipeline = hdx::createGraphicsPipeline(device, graphics_pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	camera.translate(0, 0, 3.77);
}


void Application::update(float delta_time, AppState& app_state)
{
	//if (fr_id == 1)/*
		//hdx::copyToDevice(device, accum_ssb, zero_bf, sizeof(glm::vec4) * Nx * Ny);*/

	// RayTracing computation
	hdx::recordComputeCommandBuffer(device, command_buffer, rtx_pipeline, rtx_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	hdx::beginSingleTimeCommands(device, command_buffer);
	hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
	hdx::copyBufferToImage(device, ssb, input_texture, image_width, image_height, 1, command_buffer);
	hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR32G32B32A32Sfloat, command_buffer, 1, 1);
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

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};
	vk::Buffer vbs[] = { vb_desc.buffer };
	uint64_t offsets[] = { 0 };
	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(graphics_pipeline, graphics_pipeline_layout, index_count, command_buffer, vbs, ib_desc.buffer, _DS, offsets, 1, 1);
	hdx::endRenderpass(command_buffer);

	vk::Semaphore wait_semaphores[] = { image_available_semaphore };

	vk::SubmitInfo submit_info = 0;
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &image_available_semaphore;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &render_finished_semaphore;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
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
	hdx::cleanupBuffer(device, vb_desc);
	hdx::cleanupBuffer(device, ub_desc);
	hdx::cleanupBuffer(device, ib_desc);
	hdx::cleanupBuffer(device, ssb);
	hdx::cleanupBuffer(device, accum_ssb);
	hdx::cleanupBuffer(device, ray_ssb);
	hdx::cleanupBuffer(device, ubo);
	hdx::cleanupImage(device, input_texture);
	hdx::cleanupImage(device, output_texture);
	device.destroySampler(sampler);

	device.destroyPipeline(rtx_pipeline);
	device.destroyPipelineLayout(rtx_pipeline_layout);
	device.destroyPipeline(graphics_pipeline);
	device.destroyPipelineLayout(graphics_pipeline_layout);

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