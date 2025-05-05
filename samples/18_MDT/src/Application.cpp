
#include "Application.h"


// Function to convert RGB to RGBA
unsigned char* convertRGBtoRGBA(unsigned char* rgbImage, int width, int height, unsigned char alpha) {
	int i, j;
	unsigned char* rgbaImage = (unsigned char *)malloc(width * height * 4);
	for (i = 0, j = 0; i < width * height * 3; i += 3, j += 4) {
		rgbaImage[j] = rgbImage[i];       // R
		rgbaImage[j + 1] = rgbImage[i + 1]; // G
		rgbaImage[j + 2] = rgbImage[i + 2]; // B
		rgbaImage[j + 3] = alpha;          // A (transparency)
	}

	return rgbaImage;
}


Application::Application()
{
	float aspectRatio = float(WIDTH) / float(HEIGHT);
	camera = PerspectiveCamera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
	mvp.projection = camera.getProjectionMatrix();
	mvp.view = camera.getViewMatrix();
	mvp.model = glm::mat4(1.0f);

	window = new Window("Exact Distance Transform(Compute)", WIDTH, HEIGHT);
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

	unsigned char* pixela = stbi_load("res/tile.png", &image_width, &image_height, &bytes_per_pixel, 0);
	if (!pixela)
	{
		throw std::runtime_error("Failed to load texture image!");
	}
	std::cout << "BPP : " << bytes_per_pixel;

	bytes_per_pixel = 4;

	unsigned char* pixels =	convertRGBtoRGBA(pixela, image_width, image_height, 255);

	uint64_t image_size = static_cast<uint64_t>(image_width) * image_height * bytes_per_pixel;

	hdx::createImageDesc(device, input_texture, vk::Format::eR8G8B8A8Srgb, image_width, image_height, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);
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

	dimensions = { static_cast<uint32_t>(image_width), static_cast<uint32_t>(image_height) };
	ubo = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(UBO));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ubo, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ubo, &dimensions, sizeof(UBO));
	ubo_info = hdx::createDescriptorBufferInfo(ubo, sizeof(UBO));


	ssb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, image_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ssb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ssb, pixels, image_size);
	ssb_info = hdx::createDescriptorBufferInfo(ssb, image_size);

	pool_sizes = {
		hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
		hdx::createDescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 1);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute),
		hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute)
	};
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);

	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, mvp_info, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, image_info, 1),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eStorageBuffer, ssb_info, 2),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, ubo_info, 3)
	};
	device.updateDescriptorSets(4, _WDS.data(), 0, nullptr);


	x1_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	x2_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	y1_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	y2_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	final_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	init_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);

	x1_pipeline = hdx::createComputePipeline(device, _DSL, x1_pipeline_layout, "res/shaders/x1.comp.spv");
	x2_pipeline = hdx::createComputePipeline(device, _DSL, x2_pipeline_layout, "res/shaders/x2.comp.spv");
	y1_pipeline = hdx::createComputePipeline(device, _DSL, y1_pipeline_layout, "res/shaders/y1.comp.spv");
	y2_pipeline = hdx::createComputePipeline(device, _DSL, y2_pipeline_layout, "res/shaders/y2.comp.spv");
	final_pipeline = hdx::createComputePipeline(device, _DSL, final_pipeline_layout, "res/shaders/final.comp.spv");
	init_pipeline = hdx::createComputePipeline(device, _DSL, init_pipeline_layout, "res/shaders/init.comp.spv");

	uint32_t Nx = image_width, Ny = image_height;
	Dimension block_dimension = { 32, 32, 1 };
	Dimension grid_dimension = { (Nx + block_dimension.x - 1) / block_dimension.x,
								 (Ny + block_dimension.y - 1) / block_dimension.y,
															(1)						};

	// Initialise distance map
	hdx::recordComputeCommandBuffer(device, command_buffer, init_pipeline, init_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer
	
	// forward per-row distance computation
	hdx::recordComputeCommandBuffer(device, command_buffer, x1_pipeline, x1_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// reverse per-row distance computation
	hdx::recordComputeCommandBuffer(device, command_buffer, x2_pipeline, x2_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// forward per-column distance computation
	hdx::recordComputeCommandBuffer(device, command_buffer, y1_pipeline, y1_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer

	// reverse per-column distance computation
	hdx::recordComputeCommandBuffer(device, command_buffer, y2_pipeline, y2_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer


	// final distance computation
	hdx::recordComputeCommandBuffer(device, command_buffer, final_pipeline, final_pipeline_layout, _DS, grid_dimension.x, grid_dimension.y, grid_dimension.z);
	device.resetFences({ in_flight_fence }); // Reset the fence before submission
	hdx::submitCommand(command_buffer, queue, in_flight_fence);
	// Wait for the fence to be signaled before proceeding
	device.waitForFences(in_flight_fence, VK_TRUE, UINT64_MAX);
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources }); // Reset the command buffer
	

	hdx::beginSingleTimeCommands(device, command_buffer);
		hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
		hdx::copyBufferToImage(device, ssb, input_texture, image_width, image_height, 1, command_buffer);
		hdx::transitionImageLayout(device, input_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);

	if (image_height == image_width)
		std::cout << "\n\n" << bytes_per_pixel << "\n\n";

	out_image = (uint8_t*)malloc(image_size);// new uint8_t[image_size];
	hdx::copyFromDevice(device, ssb, out_image, image_size);
	// Save the image as PNG
	int success = stbi_write_png("agenerated/output_image.png", image_width, image_height, bytes_per_pixel, out_image, image_width * bytes_per_pixel);
	if (success) {
		printf("Image saved successfully!\n");
	}
	else {
		printf("Failed to save image\n");
	}

	// Free the image memory
	stbi_image_free(pixels);

	graphics_pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	graphics_pipeline = hdx::createGraphicsPipeline(device, graphics_pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	camera.translate(0, 0, 3.77);
}


void Application::update(float delta_time, AppState& app_state)
{
	mvp.view = camera.getViewMatrix();
	hdx::copyToDevice(device, ub_desc, &mvp, sizeof(MVP));
	float rv = 0.02f * delta_time;
	float mv = 0.002f * delta_time;
	if (Input::GetKey(Input::KEY_I))
	{
		camera.rotate(-rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_J))
	{
		camera.rotate(0, rv, 0);
	}
	if (Input::GetKey(Input::KEY_K))
	{
		camera.rotate(rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_L))
	{
		camera.rotate(0, -rv, 0);
	}
	if (Input::GetKey(Input::KEY_W))
	{
		camera.translate(0, 0, mv);
	}
	if (Input::GetKey(Input::KEY_A))
	{
		camera.translate(-mv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_S))
	{
		camera.translate(0, 0, -mv);
	}
	if (Input::GetKey(Input::KEY_D))
	{
		camera.translate(mv, 0, 0);
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
	hdx::cleanupBuffer(device, ubo);
	hdx::cleanupImage(device, input_texture);
	hdx::cleanupImage(device, output_texture);
	device.destroySampler(sampler);

	device.destroyPipeline(x1_pipeline);
	device.destroyPipelineLayout(x1_pipeline_layout);
	device.destroyPipeline(x2_pipeline);
	device.destroyPipelineLayout(x2_pipeline_layout);
	device.destroyPipeline(y1_pipeline);
	device.destroyPipelineLayout(y1_pipeline_layout);
	device.destroyPipeline(y2_pipeline);
	device.destroyPipelineLayout(y2_pipeline_layout);
	device.destroyPipeline(final_pipeline);
	device.destroyPipelineLayout(final_pipeline_layout);
	device.destroyPipeline(graphics_pipeline);
	device.destroyPipelineLayout(graphics_pipeline_layout);
	device.destroyPipeline(init_pipeline);
	device.destroyPipelineLayout(init_pipeline_layout);

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