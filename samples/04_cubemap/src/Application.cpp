
#include "Application.h"


Application::Application()
{
	window = new Window("SKY", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "SKYB", enable_validation_layers, validation_layers);
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
	float aspectRatio = float(WIDTH) / float(HEIGHT);
	camera3D = PerspectiveCamera(
		glm::vec3(0.0f, 0.0f, 5.0f), // eye
		glm::vec3(0.0f, 0.0f, 0.0f), // center
		glm::radians(45.0f), // fov
		aspectRatio, //aspect ratio
		0.1f, 1000.0f); // near and far points

	mvp.projection = camera3D.getProjectionMatrix();
	mvp.view = glm::mat4(glm::mat3(camera3D.getViewMatrix()));

	device_desc.physical_device.getMemoryProperties(&device_desc.memory_properties);
	device_desc.physical_device.getProperties(&device_desc.properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;
	msaa_samples = hdx::getMaxUsableSampleCount(device_desc.physical_device);
	device_desc.physical_device.getFormatProperties(vk::Format::eR8G8B8A8Srgb, &device_desc.format_properties);

	capabilities = device_desc.physical_device.getSurfaceCapabilitiesKHR(surface);
	formats = device_desc.physical_device.getSurfaceFormatsKHR(surface);
	presentModes = device_desc.physical_device.getSurfacePresentModesKHR(surface);

	swap_chain = hdx::createSwapchain(device, surface, vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities, WIDTH, HEIGHT, extent);
	swapchain_images = device.getSwapchainImagesKHR(swap_chain);
	swapchain_size = swapchain_images.size();

	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);
	s_command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}

	for (const auto& face : faces)
	{
		stbi_uc* pixels = stbi_load(face.c_str(), &image_width, &image_height, &image_channels, STBI_rgb_alpha);
		if (!pixels)
			throw std::runtime_error("Failed to load texture image!");

		uint64_t face_size = image_width * image_height * 4; // 4 bytes per pixel (RGBA)
		all_pixels.insert(all_pixels.end(), pixels, pixels + face_size);

		stbi_image_free(pixels);
	}

	createImageDesc(device, texture, vk::Format::eR8G8B8A8Srgb, image_width, image_height, vk::SampleCountFlagBits::e1, cubemap_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_cube, 6, vk::ImageCreateFlagBits::eCubeCompatible, device_desc, 1);
	sampler = hdx::createTextureSampler(device, device_desc.properties, 6);

	tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, all_pixels.size());
	hdx::allocateBufferMemory(device, device_desc.memory_properties, tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, tb, all_pixels.data(), all_pixels.size());

	vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, sizeof(float) * vertices.size());
	hdx::allocateBufferMemory(device, device_desc.memory_properties, vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, vb, vertices.data(), sizeof(float) * vertices.size());

	ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));

	hdx::beginSingleTimeCommands(device, s_command_buffer);
		hdx::transitionImageLayout(device, texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 6);
		hdx::copyBufferToImage(device, tb, texture, image_width, image_height, 6, s_command_buffer);
		hdx::transitionImageLayout(device, texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 6);
	hdx::endSingleTimeCommands(device, s_command_buffer, command_pool, queue);

	binding_descriptions = { hdx::getBindingDescription(0, sizeof(float) * 8, vk::VertexInputRate::eVertex) };
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, 0));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, sizeof(float) * 4));
	
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment));
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);

	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapchain_size));
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, swapchain_size);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);
	_DBI = hdx::createDescriptorBufferInfo(ub, sizeof(MVP));
	_DII = hdx::createDescriptorImageInfo(texture, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI, 0));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII, 1));
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);

	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/skybox.vert.spv", "res/shaders/skybox.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
}


void Application::update(float delta_time, AppState& app_state)
{
	float rv = 0.1f * delta_time;
	float mv = 0.01f * delta_time;
	if (Input::GetKey(Input::KEY_I))
	{
		camera3D.rotate(-rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_J))
	{
		camera3D.rotate(0, rv, 0);
	}
	if (Input::GetKey(Input::KEY_K))
	{
		camera3D.rotate(rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_L))
	{
		camera3D.rotate(0, -rv, 0);
	}
	if (Input::GetKey(Input::KEY_W))
	{
		camera3D.translate(0, 0, mv);
	}
	if (Input::GetKey(Input::KEY_A))
	{
		camera3D.translate(-mv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_S))
	{
		camera3D.translate(0, 0, -mv);
	}
	if (Input::GetKey(Input::KEY_D))
	{
		camera3D.translate(mv, 0, 0);
	}

	if (Input::GetKey(Input::KEY_ESCAPE))
	{
		app_state.running = false;
	}

	mvp.view = glm::mat4(glm::mat3(camera3D.getViewMatrix()));
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));

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

	device.resetFences({ in_flight_fence });

	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });

	vk::Buffer vbs[] = { vb.buffer };
	uint64_t offsets[] = { 0 };

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};
	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(pipeline, pipeline_layout, 36, command_buffer, vbs, _DS, offsets);
	hdx::endRenderpass(command_buffer);


	vk::SubmitInfo submit_info{};
	submit_info = 0;
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &image_available_semaphore;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &render_finished_semaphore;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
	queue.submit(submit_info, in_flight_fence);

	vk::PresentInfoKHR present_info;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swap_chain;
	present_info.pImageIndices = &image_index;

	vk::Result present_result;
	try
	{
		present_result = queue.presentKHR(present_info);
	}
	catch (vk::OutOfDateKHRError error)
	{
		recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}

	current_frame++;
	current_frame = (current_frame == swapchain_size) ? 0 : current_frame;

	queue.waitIdle();
	device.waitIdle();
}



Application::~Application()
{
	cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	hdx::cleanupBuffer(device, vb);
	hdx::cleanupBuffer(device, tb);
	hdx::cleanupBuffer(device, ub);
	hdx::cleanupImage(device, texture);
	device.destroySampler(sampler);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipeline_layout);
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